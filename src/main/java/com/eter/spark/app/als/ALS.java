package com.eter.spark.app.als;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

/**
 * Created by abosii on 4/18/2017.
 */
public class ALS {
    public static void main(String[] args) {
        SparkSession session = new SparkSession.Builder()
                .appName("ALS")
                .enableHiveSupport()
                .getOrCreate();

        Dataset<Row> dataset = session.sql("SELECT * FROM productsrating");
        MatrixFactorizationModel alsModel = org.apache.spark.mllib.recommendation.ALS
                .train(transformToRDD(dataset).rdd(), 10, 10, 0.01);
    }

    public static JavaRDD<Rating> transformToRDD(Dataset<Row> dataset) {
        return dataset.javaRDD().map(new Function<Row, Rating>() {
            public Rating call(Row value) {
                int userIdIndex = value.fieldIndex("customerid");
                int productIdIndex = value.fieldIndex("productid");
                int ratingIndex = value.fieldIndex("rating");
                return new Rating((int) value.getLong(userIdIndex),
                        (int) value.getLong(productIdIndex),
                        value.getDouble(ratingIndex));
            }
        }).cache();
    }
}
