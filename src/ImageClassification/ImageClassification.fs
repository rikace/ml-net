namespace ImageClassification

open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data

[<AutoOpen>]
module CommonImageClassification =
    let imageHeight = 224
    let imageWidth = 224
    let mean = 117
    let scale = 1
    let channelsLast = true

    [<Literal>]
    let OutputTensorName = "softmax2"

    [<CLIMutable>]
    type ImageNetData =
        {
            [<LoadColumn(0)>]
            ImagePath : string
            [<LoadColumn(1)>]
            Label : string
        }

    [<CLIMutable>]
    type ImageNetPipelineTraining =
        {
            ImagePath : string
            Label : string
            PredictedLabelValue : string
            Score : float32 []
            softmax2_pre_activation : float32 []
        }

    [<CLIMutable>]
    type ImageNetDataProbability =
        {
            ImagePath : string
            Label : string
            PredictedLabel : string
            Probability : float32
        }

    [<CLIMutable>]
    type ImageNetPrediction =
        {
            ImagePath : string
            Label : string
            PredictedLabelValue : string
            Score : float32 []
        }

module Common =
    let setPath path = sprintf "%s/%s" Environment.CurrentDirectory path

    /// Helper function to cast the ML pipeline to an estimator
    let castToEstimator (x : IEstimator<_>) =
        match x with
        | :? IEstimator<ITransformer> as y -> y
        | _ -> failwith "Cannot cast pipeline to IEstimator<ITransformer>"


    let saveModel (ml:MLContext) schema model (path:string) =
        use fsWrite = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write)
        ml.Model.Save(model, schema, fsWrite)

    let printColor (color: ConsoleColor) (text: string) =
        let bakColor = Console.ForegroundColor
        Console.ForegroundColor <- color
        Console.WriteLine(sprintf "%s" text)
        Console.ForegroundColor <- bakColor

    let printfColor (color: ConsoleColor) (text: string) =
        let bakColor = Console.ForegroundColor
        Console.ForegroundColor <- color
        Console.Write(sprintf "%s" text)
        Console.ForegroundColor <- bakColor

    let printRed = printColor ConsoleColor.Red
    let printGreen = printColor ConsoleColor.Green
    let printCyan = printColor ConsoleColor.Cyan
    let printYellow = printColor ConsoleColor.Yellow
    let printBlue = printColor ConsoleColor.Blue
    let printMagenta = printColor ConsoleColor.Magenta
    let printfMagenta = printfColor ConsoleColor.Magenta
    let printn = printColor  Console.ForegroundColor
    let print = printfColor  Console.ForegroundColor

    let printExn lines =
        printfn " "
        printRed "EXCEPTION"
        printRed "#########"
        lines |> Seq.iter printn

    let printImageTraining (x : ImageNetPipelineTraining) =
        print "ImagePath: "
        printfMagenta (Path.GetFileName(x.ImagePath))
        print " predicted as "
        printfMagenta x.PredictedLabelValue
        print " with score "
        printBlue (x.Score |> Seq.max |> string)
        printfn ""

    let printImagePrediction (x : ImageNetPrediction) =
        print "ImagePath: "
        printfMagenta (Path.GetFileName(x.ImagePath))
        print " predicted as "
        printfMagenta x.PredictedLabelValue
        print " with score "

        printBlue (x.Score |> Seq.max |> string)
        printfn ""

    let printHeader lines =
        printfn " "
        lines |> Seq.iter printYellow
        let maxLength = lines |> Seq.map (fun x -> x.Length) |> Seq.max
        printYellow (String('#', maxLength))

module Training =

    let buildAndTrainModel dataLocation imagesFolder inputModelLocation imageClassifierZip =
        Common.printGreen "Read model"
        Common.printGreen (sprintf "Model location: %s" inputModelLocation)
        Common.printGreen (sprintf "Images folder: %s" imagesFolder)
        Common.printGreen (sprintf "Training file: %s" dataLocation)
        Common.printGreen (sprintf "Default parameters: image size =(%d,%d), image mean: %d" imageHeight imageWidth mean)

        let context = MLContext(seed = Nullable 1)

        let data = context.Data.LoadFromTextFile<ImageNetData>(dataLocation, hasHeader = false)
        let pipeline =
            EstimatorChain()
                .Append(context.Transforms.Conversion.MapValueToKey("LabelTokey", "Label"))
                .Append(context.Transforms.LoadImages("ImageReal", imagesFolder, "ImagePath"))
                .Append(context.Transforms.ResizeImages("ImageReal", imageWidth, imageHeight, inputColumnName = "ImageReal"))
                .Append(context.Transforms.ExtractPixels("input", "ImageReal", interleavePixelColors = channelsLast, offsetImage = float32 mean))
                .Append(context.Model.LoadTensorFlowModel(inputModelLocation).
                                     ScoreTensorFlowModel(outputColumnNames = [|"softmax2_pre_activation"|], inputColumnNames = [|"input"|], addBatchDimensionInput = true))
                .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy("LabelTokey", "softmax2_pre_activation"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabelValue","PredictedLabel"))

        Common.printHeader ["Training classification model"]

        let model = pipeline.Fit data

        let trainData = model.Transform data

        context.Data.CreateEnumerable<_>(trainData, false, true)
        |> Seq.iter Common.printImagePrediction

        Common.printHeader ["Classification metrics"]
        let metrics = context.MulticlassClassification.Evaluate(trainData, labelColumnName = "LabelTokey", predictedLabelColumnName = "PredictedLabel")

        Common.printn (sprintf "LogLoss is: %.15f" metrics.LogLoss)
        metrics.PerClassLogLoss
        |> Seq.map string
        |> String.concat " , "
        |> printfn "PerClassLogLoss is: %s"

        Common.printHeader ["Save model to local file"]
        let outFile = imageClassifierZip
        if File.Exists outFile then
            File.Delete(outFile)
        do
            use stream = File.OpenWrite(outFile)
            context.Model.Save(model, trainData.Schema, stream)
        Common.printYellow (sprintf "Model saved: %s" outFile)


module Predict =

    let printImageNetProb (x : ImageNetDataProbability) =
        Common.print "ImagePath: "
        Common.printMagenta (Path.GetFileName(x.ImagePath))
        Common.print " labeled as "
        Common.printMagenta x.Label
        Common.printfColor Console.ForegroundColor " predicted as "
        if x.Label = x.PredictedLabel then
             Common.printfColor ConsoleColor.Green x.PredictedLabel
        else
            Common.printRed x.PredictedLabel
        Common.printfColor Console.ForegroundColor " with probability "
        Common.printfColor ConsoleColor.Blue (string x.Probability)
        printfn ""

    let classifyImages dataLocation imagesFolder modelLocation =
        Common.printHeader ["Loading model"]

        let context = MLContext(seed = Nullable 1)
        let loadedModel, inputSchema =
            use stream = File.OpenRead(modelLocation)
            context.Model.Load(stream)
        Common.printCyan (sprintf "Model loaded: %s" modelLocation)

        let predictor = context.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(loadedModel)

        Common.printHeader ["Making classifications"]

        File.ReadAllLines(dataLocation)
        |> Seq.map (fun x -> let fields = x.Split '\t' in {ImagePath = Path.Combine(imagesFolder, fields.[0]); Label = fields.[1]})
        |> Seq.iter (predictor.Predict >> Common.printImagePrediction)


        let testImagePugLoaf = { ImagePath = Common.setPath "Pug.jpg"; Label = "" }
        let testImageBreadPug = { ImagePath = Common.setPath "BreadLoaf.jpg"; Label = "" }

        Common.printHeader ["Testing new images"]
        (predictor.Predict >> Common.printImagePrediction) testImagePugLoaf
        (predictor.Predict >> Common.printImagePrediction) testImageBreadPug
