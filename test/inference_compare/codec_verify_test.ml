open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module V = Codecs.Input_verify
module C = Codec_fixtures

(* A backend whose digest is the same for every payload: any two logical
   values then share a digest. Verification that trusted digests would pass
   everything under it. *)
module Collide = struct
  type state = unit

  let algorithm = C.text (Id.Digest_algorithm.of_string "collide")
  let create () = ()
  let update () _ ~pos:_ ~len:_ = ()
  let finish () = String.make 64 '0'
end

let colliding_ctx () =
  let hashers =
    match
      Codecs.Payload_hasher.Registry.register
        Codecs.Payload_hasher.Registry.empty
        (module Collide)
    with
    | Ok r -> r
    | Error _ -> failwith "register"
  in
  C.context ~hashers ~algorithm:Collide.algorithm ()

let describe = function
  | V.Equal _ -> "equal"
  | V.Different { difference = Spec_differs _; _ } -> "spec differs"
  | V.Different { difference = Element_differs { index }; _ } ->
      Printf.sprintf "element %Ld differs" index

let verify ctx ~expected ~imported_value recipe_nodes result out_spec =
  let key = C.canonical "x" in
  let t = C.boundary "engine_x" in
  let r = C.recipe "rebuild" recipe_nodes (result t) out_spec in
  let env = C.env ~boundary:[ (t, L.spec imported_value) ] () in
  ignore key;
  match Codecs.Recipe_eval.check env r with
  | Error e -> Format.printf "check: %a@." C.codec_error_pp e
  | Ok checked -> (
      let imported r =
        if Tensor_ref.equal r t then Some imported_value else None
      in
      match V.verify ctx env checked ~imported ~expected with
      | Error e -> Format.printf "verify: %a@." C.codec_error_pp e
      | Ok o -> Format.printf "%s@." (describe o))

let%expect_test "verification compares values, not digests" =
  let ctx = colliding_ctx () in
  let expected = C.integers ctx (C.i32 [ 4L ]) [ 1L; 2L; 3L; 4L ] in
  let same = C.integers ctx (C.i32 [ 4L ]) [ 1L; 2L; 3L; 4L ] in
  let corrupt = C.integers ctx (C.i32 [ 4L ]) [ 1L; 2L; 9L; 4L ] in
  Format.printf "digests collide: %b@."
    (Digest.Logical_value.equal (L.digest expected) (L.digest corrupt));
  let passthrough t = Recipe.Boundary t in
  verify ctx ~expected ~imported_value:same [] passthrough (C.i32 [ 4L ]);
  verify ctx ~expected ~imported_value:corrupt [] passthrough (C.i32 [ 4L ]);
  [%expect {|
    digests collide: true
    equal
    element 2 differs |}]

let%expect_test "reconstruction through a layout change" =
  let ctx = C.context () in
  (* The engine holds the tensor transposed; the recipe undoes that. *)
  let expected = C.integers ctx (C.i32 [ 2L; 3L ]) [ 0L; 1L; 2L; 3L; 4L; 5L ] in
  let engine = C.integers ctx (C.i32 [ 3L; 2L ]) [ 0L; 3L; 1L; 4L; 2L; 5L ] in
  let undo t =
    ignore t;
    Recipe.Conversion_result (C.node "t")
  in
  let nodes t = [ C.step_node "t" (Permute [ 1; 0 ]) [ Boundary t ] ] in
  let key = C.boundary "engine_x" in
  verify ctx ~expected ~imported_value:engine (nodes key) undo
    (C.i32 [ 2L; 3L ]);
  (* A transfer that dropped the transpose is a different value. *)
  let untransposed =
    C.integers ctx (C.i32 [ 3L; 2L ]) [ 0L; 1L; 2L; 3L; 4L; 5L ]
  in
  verify ctx ~expected ~imported_value:untransposed (nodes key) undo
    (C.i32 [ 2L; 3L ]);
  [%expect {|
    equal
    element 1 differs |}]

let%expect_test "spec mismatch is not a numeric difference" =
  let ctx = C.context () in
  let a = C.integers ctx (C.i32 [ 2L ]) [ 1L; 2L ] in
  let b = C.integers ctx (C.i64 [ 2L ]) [ 1L; 2L ] in
  (match V.first_difference ctx ~expected:a ~actual:b with
  | Ok (Some (V.Spec_differs _)) -> Format.printf "spec differs@."
  | Ok _ -> Format.printf "unexpected@."
  | Error e -> Format.printf "%a@." C.codec_error_pp e);
  [%expect {| spec differs |}]
