namespace ML.NET.FSharp

module Common =
    open System
    open System.IO
    open Microsoft.ML
    open Microsoft.ML.Data
    open Microsoft.ML.Transforms

    let setPath path = sprintf "%s/Data/%s" Environment.CurrentDirectory path

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

    let printRed = printColor ConsoleColor.Red
    let printGreen = printColor ConsoleColor.Green
    let printCyan = printColor ConsoleColor.Cyan
    let printYellow = printColor ConsoleColor.Yellow
    let printBlue = printColor ConsoleColor.Blue
    let printMagenta = printColor ConsoleColor.Magenta
    let print = printColor  Console.ForegroundColor


