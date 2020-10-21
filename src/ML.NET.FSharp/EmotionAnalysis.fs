namespace ML.NET.FSharp

module EmotionAnalysis =

    open System
    open System.IO
    open Microsoft.ML
    open Microsoft.ML.Data

    type SentimentData () =
        [<DefaultValue>]
        [<LoadColumn(0)>]
        val mutable public SentimentText :string

        [<DefaultValue>]
        [<LoadColumn(1)>]
        [<ColumnName("Label")>]
        val mutable public Sentiment :bool


    type SentimentPrediction () =
        [<DefaultValue>]
        [<ColumnName("PredictedLabel")>]
        val mutable public Prediction :bool

        [<DefaultValue>]
        val mutable public Probability :float32

    type Emotion =
    | Unhappy
    | Happy

    let trainDataPath = Common.setPath "review.csv"
    let trainDataDirPath = Common.setPath "sentimentlabels/*"
    //let trainDataDirPath = sprintf "%s/Data/sentimentlabels/*" Environment.CurrentDirectory


    let initPredictor (dataFile:string) =
        Common.printCyan "Initialize Predictor..."
        let context = new MLContext()

        // Load the data
        Common.printCyan "Load the data..."
        // load single file
        // let data = context.Data.LoadFromTextFile<SentimentData>(trainDataPath, hasHeader = false)
        // load multiple files "sentimentlabels/*"
        let data = context.Data.LoadFromTextFile<SentimentData>(trainDataDirPath, hasHeader = false)

        Common.printCyan "Training data..."
        let trainTestData = context.Data.TrainTestSplit(data, testFraction = 0.3)
        let trainData = trainTestData.TrainSet;
        let testData = trainTestData.TestSet;

        Common.printCyan "Create pipeline..."
        let pipeline =
          EstimatorChain()
            .Append(context.Transforms.Text.FeaturizeText("Features", "SentimentText"))
            .Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression())

            //.Append(context.BinaryClassification.Trainers.SdcaLogisticRegression())
            //.Append(context.BinaryClassification.Trainers.FastTree(numberOfLeaves = 50, minimumExampleCountPerLeaf = 20))

        Common.printCyan "Training the model..."
        let model = trainData |> pipeline.Fit

        // Evaluate the model
        Common.printCyan "Evaluate the model..."
        let predictions = model.Transform testData
        let metrics = context.BinaryClassification.Evaluate(predictions, "Label")

        Console.WriteLine()
        Common.printGreen (sprintf "Accuracy: %A" metrics.Accuracy)
        Common.printGreen (sprintf "AUC: %A" metrics.AreaUnderPrecisionRecallCurve)
        Common.printGreen (sprintf "F1: %A" metrics.F1Score)

        // Evaluate the model using cross-validation
        let scores = context.BinaryClassification.CrossValidate(data = data, estimator = Common.castToEstimator pipeline, numberOfFolds = 5)
        let mean = scores |> Seq.averageBy(fun x -> x.Metrics.F1Score)
        Common.printRed (sprintf "Mean cross-validated F1 score: %A" mean)

        // Use the model to make predictions
        Common.printCyan "Create Prediction Engine"
        context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model)

    // Load model from file
    let loadModel (path:string) =
        Common.printRed "Loading model..."
        if File.Exists path then
            use fsRead = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read)
            let mlReloaded = MLContext()
            let transformer, schema = mlReloaded.Model.Load(fsRead)
            mlReloaded.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(transformer)
        else initPredictor trainDataPath


    let scorePrediction sentimentText (prediction:SentimentPrediction) =
        Console.WriteLine()
        Common.printGreen (sprintf "Text : %s" sentimentText)
        Common.printGreen (sprintf "Sentiment score: %f" prediction.Probability)
        if prediction.Prediction then Emotion.Happy else Emotion.Unhappy

    let loadSentimentModel () = Common.setPath "EmotionAnalysisModel.zip" |> loadModel

    let predict (sentimentModel: PredictionEngine<SentimentData, SentimentPrediction>) sentimentText =

        let test = SentimentData()
        test.SentimentText <- sentimentText
        sentimentModel.Predict test

    let colorEmotion (emotion: Emotion) =
        match emotion with
        | Unhappy -> System.ConsoleColor.Red
        | Happy -> System.ConsoleColor.Yellow

    let print (emotion: Emotion) =
        let color = colorEmotion emotion
        Common.printColor color (sprintf "Sentiment: %O" emotion)

    let scoreSentiment (sentimentModel: PredictionEngine<SentimentData, SentimentPrediction>)  text =
        let prediction = predict sentimentModel text
        scorePrediction text prediction |> print

    let runPrediction () =
        let sentimentModel = loadSentimentModel()
        let samples = [
            "The food was bland and the service was average"
            "The food was bland and the service was below average"
            "The food was great and the service was excellent"
            "The only thing worse than the food was the terrible service"
            "I wouldn't let my dog eat here"
        ]
        samples |> List.iter (scoreSentiment sentimentModel)
