package com.eter.spark.app.als;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * Created by abosii on 4/18/2017.
 */
public class ALS {
    private static final Logger log = LoggerFactory.getLogger("ALS");

    public static void main(String[] args) throws IOException {

        if (args.length < 1) {
            log.error("Can't find argument for model output");
            log.debug("Actual arguments length: " + args.length);
            log.info("Use <application-name> path/to/model");
            return;
        }

        String output = args[0];

        log.info("Set model output as: " + output);

        SparkSession session = new SparkSession.Builder()
                .appName("ALS")
                .config("spark.sql.hive.metastore.version", "3.0.0")
                .config("spark.sql.hive.metastore.jars", "/usr/local/hadoop/share/hadoop/yarn/*:" +
                        "/usr/local/hadoop/share/hadoop/yarn/lib/*:" +
                        "/usr/local/hadoop/share/mapreduce/lib/*:" +
                        "/usr/local/hadoop/share/hadoop/mapreduce/*:" +
                        "/usr/local/hadoop/share/hadoop/common/*:" +
                        "/usr/local/hadoop/share/hadoop/hdfs/*:" +
                        "/usr//local/hadoop/etc/hadoop:" +
                        "/usr/local/hadoop/share/hadoop/common/lib/*:" +
                        "/usr/local/hadoop/share/hadoop/common/*:" +
                        "/usr/local/hive/lib/*:")
                .enableHiveSupport()
                .getOrCreate();

        Dataset<Row> dataset = session.sql("SELECT customerid, productid, rating FROM productsrating");
        MatrixFactorizationModel alsModel = org.apache.spark.mllib.recommendation.ALS
                .train(transformToRDD(dataset).rdd(), 10, 10, 0.01);
//        Pipeline pipeline = new Pipeline();
//        pipeline.setStages(new PipelineStage[] {new org.apache.spark.ml.recommendation.ALS()});
//        PipelineModel alsModel = pipeline.fit(dataset);
//        System.out.println(alsModel.recommendProducts(123, 1)[0]);
        alsModel.save(session.sparkContext(), output);
    }

    public static JavaRDD<Rating> transformToRDD(Dataset<Row> dataset) {
        return dataset.javaRDD().map((Function<Row, Rating>) value -> {
            int userIdIndex = value.fieldIndex("customerid");
            int productIdIndex = value.fieldIndex("productid");
            int ratingIndex = value.fieldIndex("rating");
            return new Rating((int) value.getLong(userIdIndex),
                    (int) value.getLong(productIdIndex),
                    value.getDouble(ratingIndex));
        }).cache();
    }
}
