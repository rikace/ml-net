namespace ML.NET.FSharp

module SpanDetection =

    open System
    open System.IO
    open Microsoft.ML
    open Microsoft.ML.Data

    /// The SpamInput class contains one single message which may be spam or ham.
    [<CLIMutable>]
    type SpamInput = {
        [<LoadColumn(0)>] [<ColumnName("Label")>] Verdict : string
        [<LoadColumn(1)>] Message : string
    }

    /// The SpamPrediction class contains one single spam prediction.
    [<CLIMutable>]
    type SpamPrediction = {
        [<ColumnName("PredictedLabel")>] IsSpam : bool
        Score : float32
        Probability : float32
    }

    /// This class describes what output columns we want to produce.
    [<CLIMutable>]
    type ToSpamHamLabel ={
        mutable Label : bool
    }

    type SpamResult =
    | Spam
    | Ham

    /// file paths to data files (assumes os = windows!)
    let dataPath = Common.setPath "spam.tsv"

    // set up a machine learning context
    let context = MLContext(seed = Nullable 1)

    // load the spam dataset in memory
    Common.printCyan "Load the data..."
    let data = context.Data.LoadFromTextFile<SpamInput>(dataPath, hasHeader = true, separatorChar = '\t')

    // use 80% for training and 20% for testing
    let partitions = context.Data.TrainTestSplit(data, testFraction = 0.2)
    let trainSet = partitions.TrainSet
    let testSet = partitions.TestSet

    // set up a training pipeline
    let pipeline =
        // build a training pipeline
        Common.printCyan "Create pipeline..."

        EstimatorChain()
            // step 1: transform the 'spam' and 'ham' values to true and false
            .Append(
                context.Transforms.CustomMapping(
                    Action<SpamInput, ToSpamHamLabel>(fun input output -> output.Label <- String.Compare(input.Verdict, "spam", true) = 0), "func"))

            // step 2: featureize the input text
            .Append(context.Transforms.Text.FeaturizeText("Features", "Message"))

            // step 3: use a stochastic dual coordinate ascent learner
            .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())


    // test the full data set by performing k-fold cross validation
    Common.printCyan "Performing cross validation:"
    let cvResults = context.BinaryClassification.CrossValidate(data = data, estimator = Common.castToEstimator pipeline, numberOfFolds = 5)

    // report the results
    Common.printCyan "Performing cross validation:"
    cvResults |> Seq.iter(fun f -> Common.printYellow (sprintf "  Fold: %i, Accuracy: %f" f.Fold f.Metrics.Accuracy))

    // train the model on the training set
    Common.printCyan "Training the model..."
    let model = partitions.TrainSet |> pipeline.Fit

    // evaluate the model on the test set
    Common.printCyan "Training the model..."
    let metrics = partitions.TestSet |> model.Transform |> context.BinaryClassification.Evaluate

    // report the results
    Common.printRed "Model metrics:"
    Common.printCyan(sprintf "  Accuracy:          %f" metrics.Accuracy)
    Common.printCyan(sprintf "  Auc:               %f" metrics.AreaUnderRocCurve)
    Common.printCyan(sprintf "  Auprc:             %f" metrics.AreaUnderPrecisionRecallCurve)
    Common.printCyan(sprintf "  F1Score:           %f" metrics.F1Score)
    Common.printCyan(sprintf "  LogLoss:           %f" metrics.LogLoss)
    Common.printCyan(sprintf "  LogLossReduction:  %f" metrics.LogLossReduction)
    Common.printCyan(sprintf "  PositivePrecision: %f" metrics.PositivePrecision)
    Common.printCyan(sprintf "  PositiveRecall:    %f" metrics.PositiveRecall)
    Common.printCyan(sprintf "  NegativePrecision: %f" metrics.NegativePrecision)
    Common.printCyan(sprintf "  NegativeRecall:    %f" metrics.NegativeRecall)

    // set up a prediction engine
    Common.printCyan "Create Prediction Engine"
    let engine = context.Model.CreatePredictionEngine model

    // create sample messages
    let messages = [
        { Message = "If you can get the new revenue projections to me by Friday, I'll fold them into the forecast."; Verdict = "" }
        { Message = "Can you attend a meeting in Atlanta on the 16th? I'd like to get the team together to discuss in-person."; Verdict = "" }
        { Message = "free price $250 weekly competition just text the word WIN to 80086 NOW"; Verdict = "" }
        { Message = "Please call our customer service representative on FREEPHONE 0808 145 4742 between 9am-11pm as you have WON a guaranteed $1000 cash or $5000 prize!"; Verdict = "" }
        { Message = "Home in 30 mins. Need anything from store?"; Verdict = "" }
    ]

    let printColor isSpam =
        match isSpam with
        | true -> System.ConsoleColor.Green
        | false -> System.ConsoleColor.Yellow


    let runPrediction () =
        // make the predictions
        Common.printCyan "Model predictions:"
        messages |> List.iter(fun m ->
            let p = engine.Predict m
            Common.printRed (if p.IsSpam then "Spam" else "Ham")
            Common.printColor (printColor p.IsSpam) (sprintf "%s - Classification %s" m.Message (if p.IsSpam then "Spam" else "Ham")))

