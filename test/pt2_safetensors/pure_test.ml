open Fixtures

let decode what = function
  | Ok x -> x
  | Error _ -> failwith ("fixture " ^ what ^ " did not decode")

let program = decode "program" (Pt2_archive.program_of_json program_json)

let archive ?(map = map_json ()) ?(weights = weights ()) () =
  let ( let* ) = Result.bind in
  let* map = Pt2_safetensors.map_of_string map in
  let weights = decode "weights" (Pt2_archive.weights_config_of_json weights) in
  Pt2_safetensors.of_parts ~map ~program ~weights
    ~constants:Pt2_archive.no_constants memory

let show label result =
  match result with
  | Ok archive -> (
      match Pt2_archive.load_captured_tensor archive "w" with
      | Error e ->
          Format.printf "%-18s load failed: %a@." label Pt2_archive.pp_error
            (Err.Error.kind e)
      | Ok t ->
          let data = t.Pt2_tensor.data in
          Format.printf "%-18s ok %a first=%ld last=%ld@." label Pt2_tensor.pp t
            (Pt2_storage.get_int32_le data 0)
            (Pt2_storage.get_int32_le data 20))
  | Error e ->
      Format.printf "%-18s %a@." label Pt2_safetensors.pp_error
        (Err.Error.kind e)

let%expect_test "a matching checkpoint loads zero-copy views" =
  show "match" (archive ());
  [%expect {| match              ok float32[2; 3] first=0 last=1084227584 |}]

let%expect_test "every disagreement is refused up front" =
  show "version" (archive ~map:(map_json ~version:2 ()) ());
  show "unmapped" (archive ~map:(map_json ~unmapped:[ "b"; "c" ] ()) ());
  show "missing key" (archive ~map:(map_json ~key:"nope" ()) ());
  show "map dtype" (archive ~map:(map_json ~dtype:"I64" ()) ());
  show "checkpoint dtype"
    (archive
       ~map:(map_json ~key:"I" ~shape:[ 3 ] ())
       ~weights:(weights ~sizes:[ 3 ] ~strides:[ 1 ] ())
       ());
  show "map shape" (archive ~map:(map_json ~shape:[ 3; 2 ] ()) ());
  show "checkpoint shape" (archive ~map:(map_json ~key:"S" ()) ());
  show "graph dtype" (archive ~weights:(weights ~dtype:5 ()) ());
  show "graph layout" (archive ~weights:(weights ~strides:[ 1; 2 ] ()) ());
  [%expect
    {|
    version            unsupported safetensors.json schema_version 2
    unmapped           the checkpoint lacks 2 captured tensor(s): b, c
    missing key        tensor "w": checkpoint has no tensor "nope"
    map dtype          tensor "w": graph dtype float32, map says I64
    checkpoint dtype   tensor "w": checkpoint dtype I64, map says F32
    map shape          tensor "w": graph shape [2; 3], map says [3; 2]
    checkpoint shape   tensor "w": checkpoint shape [4], map says [2; 3]
    graph dtype        tensor "w": graph dtype int64, map says F32
    graph layout       tensor "w": graph tensor is not a dense row-major buffer at offset 0 |}]
