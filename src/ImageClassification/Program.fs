open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open ImageClassification

[<EntryPoint>]
let main _argv =
    let assetsPath = Common.setPath "data" // pug vs BreadLoaf

    let tagsTsv = Path.Combine(assetsPath, "inputs", "data", "tags.tsv")
    let imagesFolder = Path.Combine(assetsPath, "inputs", "data")
    let inceptionPb = Path.Combine(assetsPath, "inputs", "inception", "tensorflow_inception_graph.pb")
    let imageClassifierOutputZip = Path.Combine(assetsPath, "outputs", "imageClassifier.zip")
    let imageClassifierInputZip = Path.Combine(assetsPath, "inputs", "imageClassifier.zip")

    Training.buildAndTrainModel tagsTsv imagesFolder inceptionPb imageClassifierOutputZip

    if File.Exists imageClassifierInputZip then File.Delete imageClassifierInputZip
    File.Copy(imageClassifierOutputZip, imageClassifierInputZip)

    Predict.classifyImages tagsTsv imagesFolder imageClassifierInputZip

    0
