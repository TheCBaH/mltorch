(* Host computations consume actual model outputs and checked checkpoint bytes.
   Producer host tensors are used only by the caller's comparison. *)
open Err.Syntax
open Transformers_metadata.Json_util
module D = Pt2_checkpoint_map.Dtype
module L = Pt2_fixture.Logical

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

let vector value =
  let* _ = Input.tensor value in
  let* width =
    match (value.L.dtype, value.shape) with
    | D.F32, [ 1L; width ] when width <= 1_000_000L -> Ok (Int64.to_int width)
    | _ -> invalid "host requires a bounded batch-one float32 vector"
  in
  let values = Array.init width (L.get_float value) in
  let+ () =
    Spec.require (Array.for_all Float.is_finite values) "nonfinite host vector"
  in
  values

let top5 logits =
  let* values = vector logits in
  let width = Array.length values in
  let* () = Spec.require (width >= 5) "top-five requires five classes" in
  let ids = Array.init width Fun.id in
  Array.sort
    (fun a b ->
      let order = Float.compare values.(b) values.(a) in
      if order = 0 then Int.compare a b else order)
    ids;
  (* torch.topk does not define stable tie ordering. Do not guess it. *)
  let unique = ref true in
  for i = 1 to min 5 (width - 1) do
    if values.(ids.(i - 1)) = values.(ids.(i)) then unique := false
  done;
  let* () = Spec.require !unique "top-five tied scores unsupported" in
  let* indices = Input.integers [ 1L; 5L ] (List.init 5 (Array.get ids)) in
  let+ scores =
    Input.f32 [ 1L; 5L ] (Array.init 5 (fun i -> values.(ids.(i))))
  in
  [ ("top5_ids", indices); ("top5_logits", scores) ]

let normalize features =
  let* values = vector features in
  (* The declared host route accumulates the norm in binary64 and rounds the
     norm and division to binary32. Original fixture tolerances still apply. *)
  let norm =
    round32 (sqrt (Array.fold_left (fun n x -> n +. (x *. x)) 0. values))
  in
  let* () =
    Spec.require
      (Float.is_finite norm && norm > 0.)
      "zero or nonfinite host norm"
  in
  Input.f32 features.L.shape (Array.map (fun x -> round32 (x /. norm)) values)

let clip ~log_scale ~image_features ~text_features =
  let* _ = Input.tensor log_scale in
  let* () =
    Spec.require
      (log_scale.L.dtype = D.F32 && log_scale.shape = [])
      "CLIP scale requires a float32 scalar"
  in
  let scale = round32 (exp (L.get_float log_scale 0)) in
  let* () =
    Spec.require
      (Float.is_finite scale && scale > 0.)
      "nonfinite CLIP exponential scale"
  in
  let* image = normalize image_features in
  let* text = normalize text_features in
  let* () =
    Spec.require (image.shape = text.shape) "CLIP feature widths differ"
  in
  let* i = vector image in
  let* t = vector text in
  let score = ref 0. in
  (* Match the producer's (scale * image) @ text.T operation order, with
     binary32 multiplication and an explicit binary64 dot accumulation. *)
  Array.iteri (fun n x -> score := !score +. (round32 (scale *. x) *. t.(n))) i;
  let* () = Spec.require (Float.is_finite !score) "nonfinite CLIP score" in
  let* scores = Input.f32 [ 1L; 1L ] [| !score |] in
  let+ scale = Input.f32 [] [| scale |] in
  (* Published host tensor order, retained for complete-set comparison. *)
  [
    ("image_features", image_features);
    ("text_features", text_features);
    ("image_embeds", image);
    ("text_embeds", text);
    ("logit_scale", scale);
    ("logits_per_image", scores);
    ("logits_per_text", scores);
  ]

let checkpoint_scale (fixture : Pt2_fixture_unix.Fixture.t) =
  let matches =
    Schema_runtime.String_map.bindings fixture.document.tensors
    |> List.filter (fun (_, entry) ->
        match entry.Pt2_checkpoint_map.Document.Entry.origin with
        | Checkpoint origin -> origin.key = "logit_scale"
        | Empty | Fill _ | Inline _ | Pack _ -> false)
  in
  let* target =
    match matches with
    | [ (target, _) ] -> Ok target
    | _ -> invalid "CLIP requires exactly one checkpoint logit_scale capture"
  in
  let* tensor = Pt2_archive.load_captured_tensor fixture.archive target in
  L.of_pt2 tensor |> Err.map_error (fun e -> `Logical_tensor (target, e))
