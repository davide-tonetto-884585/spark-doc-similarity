from itertools import combinations
import numpy as np
from time import time
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import regexp_replace, col, size, udf, monotonically_increasing_id
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("Similar documents") \
    .config("spark.master", "spark://127.0.0.1:7077") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.driver.memory", "22g") \
    .config("spark.executor.memory", "22g") \
    .getOrCreate()
sc = spark.sparkContext


def main():
    num_cores = 1
    for m in [64, 128, 256]:
        for p in [16, 32]:
            if m == 256 and p == 16:
                continue

            stats(num_cores, m, p)


def stats(num_cores, m, p):
    global spark, sc

    # import dataset from parquet file
    dataset = spark.read.parquet("data/mail_1.parquet")

    # dataset = spark.read.csv("data/news.csv", header=True).select("summary")
    # dataset = dataset.withColumnRenamed("summary", "text")

    # create a sample dataset
    """dataset = spark.createDataFrame([
        (0, "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam ac ante nec nunc fermentum fermentum."),
        (1, "hello spark alpha beta pppp pppp aaa "),
        (2, "hello spark alpha beta pppp pppp aaa ")
    ], ["id", "text"])"""
    # dataset = dataset.limit(10000)

    n = dataset.count()

    # clean and tokenize the text using Spark MLlib
    dataset = dataset.withColumn("text", regexp_replace(col("text"), "[^a-zA-Z\\s]", ""))

    tokenizer = Tokenizer(inputCol='text', outputCol='words')
    df = tokenizer.transform(dataset)

    del dataset

    # remove empty terms ''
    filter_udf = udf(lambda words: [word for word in words if len(word) >= 1], ArrayType(StringType()))
    df = df.withColumn('words', filter_udf(col('words')))

    # remove stopwords
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    df = remover.transform(df)
    df.drop("words")

    # remove documents without words
    df = df.filter(size(col("filtered")) > 0)

    # count vectorizer
    cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")  # , minDF=0.01)
    cv_model = cv.fit(df)
    df = cv_model.transform(df)
    df.drop("filtered")

    print("Number of words: " + str(len(cv_model.vocabulary)))

    idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
    idf_model = idf.fit(df)
    df = idf_model.transform(df)
    df.drop("rawFeatures")

    # add an id column to the dataset with progressive number
    df = df.withColumn('id', monotonically_increasing_id())

    # create a random matrix with {-1, +1} values of size m x n_words as a broadcast variable
    random_matrix = sc.broadcast(np.random.choice([-1, 1], (m, len(cv_model.vocabulary))))

    # Define simhash function inside main where broadcast variables are accessible
    def simhash(features):
        # Compute SimHash
        sim_hash = np.sum(
            [(random_matrix.value[:, i] * features[int(i)]) for i in features.indices],
            axis=0
        )

        # Check if the simHash is not a list
        if not isinstance(sim_hash, np.ndarray):
            return [0] * m

        # Return the simHash as a list of 0s and 1s
        return (sim_hash >= 0).astype(int).tolist()

    # create a UDF to compute the simHash
    simhash_udf = udf(simhash, ArrayType(IntegerType()))

    # time the computation
    print("\nComputing the simHash...")
    start = time()

    # compute the simHash for each document
    df = df.withColumn("simHash", simhash_udf('features'))
    df.persist()
    df.collect()

    _time_simhash = time() - start
    print("Time to calculate the simHashes: " + str(_time_simhash))

    with open(f"./result/simhash_time.csv", "a") as f:
        f.write(f"{num_cores},{m},{p},{str(_time_simhash)}" + "\n")

    # divide simHash in blocks of p elements and convert each block to an integer
    blocks_udf = udf(lambda simHash: [int("".join(map(str, simHash[i:i + p])), 2) for i in range(0, m, p)],
                     ArrayType(IntegerType()))
    df = df.withColumn("simHashBlocks", blocks_udf('simHash'))

    # show the dataset
    # df.show()

    # group the simHashes that shares at least one block
    print("\nGrouping similar documents...")
    start = time()

    # create dataframe with the (id, simHashBlock, position) for each row and group by simHashBlock and position
    block_dataframe = df.select("id", "simHash", "simHashBlocks").rdd.flatMap(
        lambda x: [((x[0], x[1]), block, i) for i, block in enumerate(x[2])]).toDF(["id", "block", "position"])
    grouped_blocks = block_dataframe.groupBy("block", "position").agg({"id": "collect_list"})

    del df

    grouped_blocks.persist()
    grouped_blocks.collect()

    _time_blocks = time() - start
    print("Time to group the blocks: " + str(_time_blocks))

    with open(f"./result/group_blocks_time.csv", "a") as f:
        f.write(f"{num_cores},{m},{p},{str(_time_blocks)}" + "\n")

    # show the grouped blocks
    # grouped_blocks.show()

    # sum len of lists with more than 1 element
    tot_comparison = int((n ** 2 - n) / 2)
    group_comparison = int(grouped_blocks.filter(size(col("collect_list(id)")) > 1).rdd.map(
        lambda x: (len(x[2]) ** 2 - len(x[2])) / 2).sum())

    print("Percentage of comparisons to do: " + str(group_comparison / tot_comparison * 100) + "%")

    with open(f"./result/comparisons.csv", "a") as f:
        f.write(f"{num_cores},{m},{p},{str(group_comparison / tot_comparison)}" + "\n")

    # find similar documents efficiently by comparing the simHashes that share at least one block
    print("\nFinding similar documents...")

    _time = time()

    docs_to_compare = grouped_blocks.select("collect_list(id)").rdd \
        .flatMap(lambda x: combinations(x[0], 2)).map(
        lambda x: (x[0][0], x[1][0], [x[0][1], x[1][1]])) \
        .toDF(["id1", "id2", "simHashes"]).dropDuplicates(["id1", "id2"])

    similar_docs_rdd = docs_to_compare.rdd \
        .map(lambda x: (x[0], x[1], float(1 - np.sum([1 for i, j in zip(x[2][0], x[2][1]) if i != j]) / m))) \
        .filter(lambda x: x[2] >= 0.9)

    similar_docs = similar_docs_rdd.collect()

    _time_similarity = time() - _time
    print("Time to find similar documents:", _time_similarity)

    with open(f"./result/similar_docs_time.csv", "a") as f:
        f.write(f"{num_cores},{m},{p},{str(_time_similarity)}" + "\n")

    with open(f"./result/num_similar_docs.csv", "a") as f:
        f.write(f"{num_cores},{m},{p},{str(len(similar_docs))}" + "\n")

    # show the similar documents
    print("\nNumber of similar docs: " + str(len(similar_docs)))


if __name__ == "__main__":
    main()

    # stop the spark session
    spark.stop()
