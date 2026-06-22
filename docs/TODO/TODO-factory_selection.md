Factory selection: add ModelFlavor detect_from_metadata(
const ModelMetadata&); rather than relying entirely on stringmatching. For example:

input:
[1,T]

output:
[1,192]

might strongly suggest:

SpeechBrain

while:

input:
[1,T,80]

suggests:

WeSpeaker-style export

This becomes useful when somebody renames:

speechbrain.onnx

to:

speaker.onnx

and your string matching stops working. The best solution would be metadata-driven loading so that future ECAPA exports can be supported without adding new filename heuristics.
