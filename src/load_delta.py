from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import logging


def load_to_delta(input_path: str, delta_base_path: str):
    """
    Appends enriched daily row for each stock into stock-specific delta tables.
    Reads existing delta table schema to ensure compatibility.
    
    Args:
        input_path: Path to enriched parquet file
        delta_base_path: Base path for delta tables (e.g., 'delta_tables')
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("[DELTA] Starting Spark session...")
    builder = (
        SparkSession.builder
        .appName("stock_delta_append")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        # Read enriched parquet data
        logger.info(f"[DELTA] Reading enriched data from {input_path}")
        df = spark.read.parquet(input_path)
        
        if df.count() == 0:
            logger.warning("[DELTA] No data to append. Exiting.")
            return
        
        # Process each stock symbol
        symbols = df.select("stock_symbol").distinct().collect()
        
        for row in symbols:
            sym = row["stock_symbol"]
            target = f"{delta_base_path}/stock_{sym}"
            
            logger.info(f"[DELTA] Processing {sym} → {target}")
            
            try:
                # Filter data for this symbol
                sym_df = df.filter(col("stock_symbol") == sym)
                
                # Read existing delta table to get schema
                existing_df = spark.read.format("delta").load(target)
                existing_schema = existing_df.schema
                existing_columns = [field.name for field in existing_schema.fields]
                
                logger.info(f"[DELTA] Existing schema columns: {existing_columns}")
                logger.info(f"[DELTA] New data columns: {sym_df.columns}")
                
                # Print schemas for debugging
                logger.info(f"[DELTA] Existing schema:")
                existing_df.printSchema()
                logger.info(f"[DELTA] New data schema:")
                sym_df.printSchema()
                
                # Cast new data to match existing schema types
                aligned_df = sym_df
                for field in existing_schema.fields:
                    if field.name in aligned_df.columns:
                        # Cast to the exact type from existing schema
                        aligned_df = aligned_df.withColumn(field.name, col(field.name).cast(field.dataType))
                
                # Select columns in the same order as existing schema
                aligned_df = aligned_df.select(*existing_columns)
                
                # Verify we have exactly 1 row per stock
                row_count = aligned_df.count()
                logger.info(f"[DELTA] Appending {row_count} row(s) to {target}")
                
                if row_count != 1:
                    logger.warning(f"[DELTA] Expected 1 row for {sym}, got {row_count}")
                
                # Append to delta table
                aligned_df.write.format("delta").mode("append").save(target)
                
                logger.info(f"[DELTA] ✅ Successfully appended data for {sym}")
                
            except Exception as e:
                logger.error(f"[DELTA] ❌ Failed to append data for {sym}: {e}")
                raise  # Fail entire task if one stock fails
        
        logger.info("[DELTA] ✅ Completed append for all stocks")
        
    finally:
        logger.info("[DELTA] Closing Spark session...")
        spark.stop()
