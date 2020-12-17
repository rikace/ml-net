open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms
open ML.NET.FSharp


[<EntryPoint>]
let main argv =

    RecommendationEngine.runPrediction()


    // DemoCarPricePrediction.runPrediction()
    // CarPricePrediction.runPrediction()
    // SpanDetection.runPrediction()
    // CharacterRecognition.runPrediction()
    // NetflixMovieRecommender.runPrediction()


    // EXTRAS
    // ElectricBikeRentPrediction.runPrediction()
    // AutoPrediction.runPrediction()
    // EmotionAnalysis.runPrediction()
    // IrisClassification.runPrediction()
    // TitanicPrediction.runPrediction()
    // HousePricePrediction.runMedianHouseValue()
    // HousePricePrediction.runLocation()

    printfn "Program Completed"
    Console.ReadLine() |> ignore
    0


