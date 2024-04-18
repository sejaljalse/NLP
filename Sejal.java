java
import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;

public class SentimentAnalysis {

public static void main(String[] args) throws Exception {
// Load dataset
Instances data = DataSource.read(“path/to/your/dataset.arff”);
data.setClassIndex(data.numAttributes() – 1);

// Initialize Naïve Bayes classifier
Classifier nbClassifier = new NaiveBayes();
nbClassifier.buildClassifier(data);

// Evaluate classifier using cross-validation
Evaluation eval = new Evaluation(data);
eval.crossValidateModel(nbClassifier, data, 10, new Random(1));

// Print evaluation results
System.out.println(eval.toSummaryString(“\nResults\n======\n”, false));

// Make predictions
Instance testInstance = data.instance(0); // Example test instance
double prediction = nbClassifier.classifyInstance(testInstance);
System.out.println(“Predicted class: “ + data.classAttribute().value((int) prediction));
}
}
