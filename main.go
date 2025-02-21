package main

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

type Result struct {
	DistMetric string
	K          int
	Acc        float64
}

func main() {

	defer lg.Sync()

	// Загрузка данных
	data, err := base.ParseCSVToInstances("./samples/knn_go.csv", true)
	if err != nil {
		lg.DPanicln("Failed to import CSV file")
	}

	lg.Infoln("CSV file imported successfully")

	// Разделение данных
	trainData, testData := base.InstancesTrainTestSplit(data, 0.7)

	Results := make([]Result, 1000000)
	metrics := []string{"euclidean", "manhattan", "cosine"}
	kValues := []int{3, 5, 7, 9, 11, 13, 15, 17}
	for _, metric := range metrics {
		for _, k := range kValues {
			cls := knn.NewKnnClassifier(metric, "linear", k)
			cls.Fit(trainData)
			predictions, _ := cls.Predict(testData)
			confusionMat, _ := evaluation.GetConfusionMatrix(testData, predictions)
			acc := evaluation.GetAccuracy(confusionMat)
			// fmt.Printf("Метрика: %s, k=%d → Точность: %.2f\n", metric, k, acc)
			var res = Result{
				DistMetric: metric,
				K:          k,
				Acc:        acc,
			}
			Results = append(Results, res)
		}
	}
	bestRes := Result{}
	bestAcc := 0.0

	for _, res := range Results {
		if res.Acc > bestAcc {
			bestAcc = res.Acc
			bestRes = res
		}
	}
	fmt.Printf("Dist Metric: %s K: %d Acc: %f\n", bestRes.DistMetric, bestRes.K, bestRes.Acc)
}
