namespace ML.NET.FSharp

module ElectricBikeRentPrediction =
    open System
    open Microsoft.ML
    open Microsoft.ML.Data

    /// The DemandObservation class holds one single bike demand observation record.
    [<CLIMutable>]
    type InputPrediction = {
        [<LoadColumn(2)>] Season : float32
        [<LoadColumn(3)>] Year : float32
        [<LoadColumn(4)>] Month : float32
        [<LoadColumn(5)>] Hour : float32
        [<LoadColumn(6)>] Holiday : float32
        [<LoadColumn(7)>] Weekday : float32
        [<LoadColumn(8)>] WorkingDay : float32
        [<LoadColumn(9)>] Weather : float32
        [<LoadColumn(10)>] Temperature : float32
        [<LoadColumn(11)>] NormalizedTemperature : float32
        [<LoadColumn(12)>] Humidity : float32
        [<LoadColumn(13)>] Windspeed : float32
        [<LoadColumn(16)>] [<ColumnName("Label")>] Count : float32
    }

    /// The DemandPrediction class holds one single bike demand prediction.
    [<CLIMutable>]
    type OutputPrediction = {
        [<ColumnName("Score")>] PredictedCount : float32;
    }

    // file paths to data files (assumes os = windows!)
    let dataPath = Common.setPath "electric_bike_demand.csv"

    let runPrediction () =

        // create the machine learning context
        let context = new MLContext();

        // load the dataset
        let data = context.Data.LoadFromTextFile<InputPrediction>(dataPath, hasHeader = true, separatorChar = ',')

        // split the dataset into 80% training and 20% testing
        let partitions = context.Data.TrainTestSplit(data, testFraction = 0.2)

        // build a training pipeline
        let pipeline =
            EstimatorChain()

                // step 1: concatenate all feature columns
                .Append(context.Transforms.Concatenate("Features", "Season", "Year", "Month", "Hour", "Holiday", "Weekday", "WorkingDay", "Weather", "Temperature", "NormalizedTemperature", "Humidity", "Windspeed"))

                // step 2: cache the data to speed up training
                .AppendCacheCheckpoint(context)

                // step 3: use a fast forest learner
                .Append(context.Regression.Trainers.FastForest(numberOfLeaves = 20, numberOfTrees = 100, minimumExampleCountPerLeaf = 10))

        // train the model
        let model = partitions.TrainSet |> pipeline.Fit

        // evaluate the model
        let metrics = partitions.TestSet |> model.Transform |> context.Regression.Evaluate

        // show evaluation metrics
        Common.printGreen "Model metrics:"
        Common.printGreen (sprintf "  RMSE:%f" metrics.RootMeanSquaredError)
        Common.printGreen (sprintf "  MSE: %f" metrics.MeanSquaredError)
        Common.printGreen (sprintf "  MAE: %f" metrics.MeanAbsoluteError)

        // set up a sample observation
        let sample1 = {
            Season = 3.0f
            Year = 1.0f
            Month = 8.0f
            Hour = 10.0f
            Holiday = 0.0f
            Weekday = 4.0f
            WorkingDay = 1.0f
            Weather = 1.0f
            Temperature = 0.8f
            NormalizedTemperature = 0.7576f
            Humidity = 0.55f
            Windspeed = 0.2239f
            Count = 0.0f // the field to predict
        }

        let sample2 = {
            Season = 1.0f
            Year = 1.0f
            Month = 2.0f
            Hour = 8.0f
            Holiday = 0.0f
            Weekday = 4.0f
            WorkingDay = 1.0f
            Weather = 0.0f
            Temperature = 0.4f
            NormalizedTemperature = 0.43f
            Humidity = 0.75f
            Windspeed = 0.4239f
            Count = 0.0f // the field to predict
        }

        // create a prediction engine
        let engine = context.Model.CreatePredictionEngine model

        for sample in [sample1;sample2] do
            // make the prediction
            let prediction = sample |> engine.Predict

            // show the prediction
            Common.printYellow "\r"
            Common.printYellow (sprintf "Single prediction for month %f :" sample.Month)
            Common.printYellow (sprintf "  Predicted bike count: %f" prediction.PredictedCount)

