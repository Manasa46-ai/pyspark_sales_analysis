from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, max, min

# Step 1: Create Spark Session
spark = SparkSession.builder \
    .appName("SalesDataAnalysis") \
    .getOrCreate()

# Step 2: Load Dataset
df = spark.read.csv("sales.csv", header=True, inferSchema=True)

print("Schema:")
df.printSchema()

print("Sample Data:")
df.show(5)

# Step 3: Data Cleaning
df = df.dropna().dropDuplicates()

# Step 4: Add Revenue Column
df = df.withColumn("revenue", col("quantity") * col("price"))

# Step 5: Analysis
print("\n🔹 Total Sales by Category")
df.groupBy("category") \
    .agg(sum("revenue").alias("total_revenue")) \
    .orderBy(col("total_revenue").desc()) \
    .show()

print("\n🔹 Average Order Value by Region")
df.groupBy("region") \
    .agg(avg("revenue").alias("avg_revenue")) \
    .orderBy(col("avg_revenue").desc()) \
    .show()

print("\n🔹 Top 3 Best Selling Products")
df.groupBy("product") \
    .agg(sum("quantity").alias("total_quantity")) \
    .orderBy(col("total_quantity").desc()) \
    .show(3)

print("\n🔹 Min & Max Revenue Transactions")
df.select(min("revenue").alias("min_revenue"),
          max("revenue").alias("max_revenue")).show()

# Step 6: Save Results
df.write.mode("overwrite").parquet("output/sales_processed")

print("\n✅ Analysis Complete. Results saved in 'output/sales_processed'")

# Stop Spark session
spark.stop()
