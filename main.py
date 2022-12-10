import pyspark.sql.functions as f
from graphframes import GraphFrame
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, StringType, DoubleType

spark = SparkSession.builder.appName("YoutubeAnalyser").getOrCreate()

schema = StructType() \
    .add("video_id", StringType(), True) \
    .add("uploader", StringType(), True) \
    .add("age", IntegerType(), True) \
    .add("category", StringType(), True) \
    .add("length", IntegerType(), True) \
    .add("views", IntegerType(), True) \
    .add("rate", DoubleType(), True) \
    .add("ratings", IntegerType(), True) \
    .add("comments", IntegerType(), True) \
    .add("ref1", StringType(), True) \
    .add("ref2", StringType(), True) \
    .add("ref3", StringType(), True) \
    .add("ref4", StringType(), True) \
    .add("ref5", StringType(), True) \
    .add("ref6", StringType(), True) \
    .add("ref7", StringType(), True) \
    .add("ref8", StringType(), True) \
    .add("ref9", StringType(), True) \
    .add("ref10", StringType(), True) \
    .add("ref11", StringType(), True) \
    .add("ref12", StringType(), True) \
    .add("ref13", StringType(), True) \
    .add("ref14", StringType(), True) \
    .add("ref15", StringType(), True) \
    .add("ref16", StringType(), True) \
    .add("ref17", StringType(), True) \
    .add("ref18", StringType(), True) \
    .add("ref19", StringType(), True) \
    .add("ref20", StringType(), True)

df = spark.read.format("csv") \
    .option("header", True) \
    .option("delimiter", "\t") \
    .schema(schema) \
    .option("inferSchema", "true") \
    .load("1.txt")

video_length = StructType() \
    .add("id", StringType(), True) \
    .add("length", IntegerType(), True)

video_size = StructType() \
    .add("id_", StringType(), True) \
    .add("length_", IntegerType(), True) \
    .add("size", IntegerType(), True)

dfVideoLength = spark.read.format("csv") \
    .option("header", True) \
    .option("delimiter", "\t") \
    .schema(video_length) \
    .option("inferSchema", "true") \
    .load("idlength.txt")

dfVideoSize = spark.read.format("csv") \
    .option("header", True) \
    .option("delimiter", "\t") \
    .schema(video_size) \
    .option("inferSchema", "true") \
    .load("size.txt")

dfMappedLength = dfVideoLength.join(dfVideoSize, dfVideoSize.id_ == dfVideoLength.id).select('id', 'length_', 'size')

dfRawData = df.select("video_id", "uploader", "age", "category", "length", "views", "rate", "ratings", "comments")

dfSIzeMappedWithRaw = dfMappedLength.join(dfRawData, dfRawData.video_id == dfMappedLength.id).select('*')
dfSIzeMappedWithRaw.show()

columns = [f.col('ref1'), f.col('ref2'), f.col('ref3'), f.col('ref4'), f.col('ref5'), f.col('ref6'), f.col('ref7'),
           f.col('ref8'), f.col('ref9'), f.col('ref10'), f.col('ref11'), f.col('ref12'), f.col('ref13'), f.col('ref14'),
           f.col('ref15'), f.col('ref16'), f.col('ref17'), f.col('ref18'), f.col('ref19'), f.col('ref20')]

output = df.withColumn("related", f.array(columns)).select("video_id", "related")

# output.printSchema()
dfEdge = output.select(output.video_id, f.explode(output.related))
dfEdge = dfEdge.withColumnRenamed("col", "dst") \
    .withColumnRenamed("video_id", "src")
# dfEdge.printSchema()

# A. Network aggregation

g = GraphFrame(dfSIzeMappedWithRaw, dfEdge)
# MAX indegree
g.inDegrees.agg({"inDegree": "max"}).show()
# MIN indegree
g.inDegrees.agg({"inDegree": "min"}).show()
# AVG indegree
g.inDegrees.agg({"inDegree": "avg"}).show()

g.outDegrees.show()
# MAX outdegree
g.outDegrees.agg({"outDegree": "max"}).show()
# MIN outdegree
g.outDegrees.agg({"outDegree": "min"}).show()
# AVG outdegree
g.outDegrees.agg({"outDegree": "avg"}).show()

# B. Search
# - top k queries
g.vertices.groupBy("category").count().sort(f.col("count").desc()).show()
# dfVertex.groupby('category').count().sort(f.col("count").desc()).show()

# Range queries: find all videos in categories X with duration within a range [t1, t2]; find all
g.vertices.filter((f.col('length').between(0, 9999999999)) & (f.col('category') == 'Sports')).show()

# videos with size in range [x,y].
g.vertices.groupBy("uploader").count().sort(f.col("count").desc()).show()
g.vertices.filter((f.col('size').between(0, 9999999999)) & (f.col('category') == 'Sports')).show()


# C. PageRank
pr = g.pageRank(resetProbability=0.15, tol=0.01)
pr.vertices.show()
pr.edges.show()
