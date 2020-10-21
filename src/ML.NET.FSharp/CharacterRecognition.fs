namespace ML.NET.FSharp

module CharacterRecognition =

    open System
    open System.IO
    open Microsoft.ML
    open Microsoft.ML.Data
    open Microsoft.ML.Transforms

    /// The Digit class represents one mnist digit.
    [<CLIMutable>]
    type Digit = {
        [<LoadColumn(0, 63)>] [<VectorType(64)>] PixelValues : float32[]
        [<LoadColumn(64)>] [<ColumnName("Label")>] Digit : int
    }

    /// The DigitPrediction class represents one digit prediction.
    [<CLIMutable>]
    type DigitPrediction = {
        [<ColumnName("Score")>] Scores : float32[]
        [<ColumnName("PredictedLabel")>] Digit : int
    }

    /// file paths to train and test data files (assumes os = windows!)
    let trainDataPath = Common.setPath "mnist-digits-train.csv"
    let testDataPath = Common.setPath "mnist-digits-test.csv"


    // create a machine learning context
    let context = new MLContext(seed = Nullable 1)

    // load the datafiles
    Common.printCyan "Load the data..."
    let trainData = context.Data.LoadFromTextFile<Digit>(trainDataPath, hasHeader = true, separatorChar = ',')
    let testData = context.Data.LoadFromTextFile<Digit>(testDataPath, hasHeader = true, separatorChar = ',')

    let runPrediction () =
        // build a training pipeline
        Common.printCyan "Create pipeline..."
        let pipeline =
            EstimatorChain()

                // step 1: map the number column to a key value and store in the label column
                .Append(context.Transforms.Conversion.MapValueToKey("Label", keyOrdinality = ValueToKeyMappingEstimator.KeyOrdinality.ByValue))

                // step 2: concatenate all feature columns
                .Append(context.Transforms.Concatenate("Features", "PixelValues"))

                // step 3: cache data to speed up training
                .AppendCacheCheckpoint(context)

                // step 4: train the model with SDCA
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())

                // step 5: map the label key value back to a number
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))

        // train the model
        Common.printCyan "Training the model..."
        let model = trainData |> pipeline.Fit

        // get predictions and compare them to the ground truth
        Common.printCyan "Evaluate the model..."
        let metrics = testData |> model.Transform |> context.MulticlassClassification.Evaluate

        // show evaluation metrics
        Common.printGreen (sprintf "Evaluation metrics")
        Common.printGreen (sprintf "  MacroAccuracy:    %f" metrics.MacroAccuracy)
        Common.printGreen (sprintf "  LogLoss:          %f" metrics.LogLoss)
        Common.printGreen (sprintf "  LogLossReduction: %f" metrics.LogLossReduction)

        // grab five digits from the test data
        let digits : Digit [] = context.Data.CreateEnumerable(testData, reuseRowObject = false) |> Array.ofSeq
        let testDigits = [ digits.[5]; digits.[16]; digits.[28]; digits.[63]; digits.[129] ]

        // create a prediction engine
        Common.printCyan "Create Prediction Engine"
        let engine :PredictionEngine<Digit, DigitPrediction> = context.Model.CreatePredictionEngine model

        // show predictions
        Common.printRed "Model predictions:"
        printf "  #\t\t"; [0..9] |> Seq.iter(fun i -> printf "%i\t\t" i); printfn ""
        testDigits |> Seq.iter(
            fun digit ->
                printf "  %i\t" (int digit.Digit)
                let p = engine.Predict digit
                p.Scores |> Seq.iter (fun s -> printf "%f\t" s)
                printfn "")


        let input4 : Digit =  {
                Digit = 0
                PixelValues =
                [|
                    0.f; 0.f;  1.f;  0.f; 12.f;  2.f; 0.f; 0.f;
                    0.f; 0.f;  0.f;  6.f; 14.f;  1.f; 0.f; 0.f;
                    0.f; 0.f;  4.f; 16.f;  7.f;  8.f; 0.f; 0.f;
                    0.f; 0.f; 13.f; 10.f;  0.f; 16.f; 6.f; 0.f;
                    0.f; 3.f; 16.f; 10.f; 12.f; 16.f; 0.f; 0.f;
                    0.f; 0.f;  4.f; 10.f; 13.f; 16.f; 0.f; 0.f;
                    0.f; 0.f;  0.f;  0.f;  6.f; 16.f; 0.f; 0.f;
                    1.f; 0.f;  0.f;  0.f; 12.f;  8.f; 0.f; 0.f
                |]
        } // 4

        let input1 : Digit =  {
                Digit = 0
                PixelValues =
                [|
                   0.f; 0.f;  0.f;  0.f; 14.f; 13.f; 1.f; 0.f
                   0.f; 0.f;  0.f;  5.f; 16.f; 16.f; 2.f; 0.f
                   0.f; 0.f;  0.f; 14.f; 16.f; 12.f; 0.f; 0.f
                   0.f; 1.f; 10.f; 16.f; 16.f; 12.f; 0.f; 0.f
                   0.f; 3.f; 12.f; 14.f; 16.f;  9.f; 0.f; 0.f
                   0.f; 0.f;  0.f;  5.f; 16.f; 15.f; 0.f; 0.f
                   0.f; 0.f;  0.f;  4.f; 16.f; 14.f; 0.f; 0.f
                   0.f; 0.f;  0.f;  1.f; 13.f; 16.f; 1.f; 0.f
                |]
        } // 1

        Common.printRed "Passing pixel values for number 1 - predictions:"
        let prediction1 = engine.Predict input1
        for (i, score) in prediction1.Scores |> Seq.indexed do
            printfn "%d - %A" i score
        Common.printRed (sprintf "Looks like a %A" prediction1.Digit)


        Common.printRed "Passing pixel values for number 4 - predictions:"
        let prediction4 = engine.Predict input4
        for (i, score) in prediction4.Scores |> Seq.indexed do
            printfn "%d - %A" i score
        Common.printRed (sprintf "Looks like a %A" prediction4.Digit)
