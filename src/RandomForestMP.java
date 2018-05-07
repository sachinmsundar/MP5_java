import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.tree.RandomForest;


import java.util.HashMap;
import java.util.regex.Pattern;

public final class RandomForestMP {
	//TODO
    private static class ParsePoint implements Function<String, Vector> {
        private static final Pattern SPACE = Pattern.compile(",");

        public Vector call(String line) {

            String[] token = SPACE.split(line);
            double[] dat = new double[token.length-1];

            for (int i = 0; i < token.length - 1; ++i) {
                dat[i] = Double.parseDouble(token[i]);
            }

            return Vectors.dense(dat);
        }
    }
    private static class DataToPoint implements Function<String, LabeledPoint> {
        private static final Pattern SPACE = Pattern.compile(",");

        public LabeledPoint call(String line) throws Exception {

            String[] token = SPACE.split(line);
            double cl = Double.parseDouble(token[token.length-1]);
            double[] dat = new double[token.length-1];

            for (int i = 0; i < token.length - 1; ++i) {
                dat[i] = Double.parseDouble(token[i]);
            }

            return new LabeledPoint(cl, Vectors.dense(dat));
        }
    }
    //

    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println(
                    "Usage: RandomForestMP <training_data> <test_data> <results>");
            System.exit(1);
        }
        String training_data_path = args[0];
        String test_data_path = args[1];
        String results_path = args[2];

        SparkConf sparkConf = new SparkConf().setAppName("RandomForestMP");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
	final RandomForestModel model;

        Integer numClasses = 2;
        HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        Integer numTrees = 3;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        Integer maxDepth = 5;
        Integer maxBins = 32;
        Integer seed = 12345;

		// TODO

        JavaRDD<LabeledPoint> training_set = sc.textFile(training_data_path).map(new DataToPoint());
        JavaRDD<Vector> testing_set = sc.textFile(test_data_path).map(new ParsePoint());

        model = RandomForest.trainClassifier(training_set,
                numClasses,
                categoricalFeaturesInfo,
                numTrees,
                featureSubsetStrategy,
                impurity,
                maxDepth,
                maxBins,
                seed);

        JavaRDD<LabeledPoint> results = testing_set.map(new Function<Vector, LabeledPoint>() {
            public LabeledPoint call(Vector points) {
                return new LabeledPoint(model.predict(points), points);
            }
        });

        results.saveAsTextFile(results_path);

        sc.stop();
    }
}
