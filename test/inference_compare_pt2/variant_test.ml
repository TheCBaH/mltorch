(* The spellings of a variant: a graph view, with a configuration only where
   the engine has one. *)

module V = Inference_compare_pt2.Variant

let%expect_test "spellings parse and print canonically" =
  List.iter
    (fun s ->
      match V.of_string s with
      | Ok v -> Printf.printf "%s -> %s\n" s (V.to_string v)
      | Error m -> Printf.printf "%s -> error: %s\n" s m)
    [
      "aten";
      "native";
      "native.canonical";
      "native4d";
      "native:sequential32";
      "native.canonical:sequential32";
      "native-sequential";
      "native4d:sequential32";
      "aten:sequential32";
      "native:fast";
      "wasm";
    ];
  [%expect
    {|
    aten -> aten
    native -> native
    native.canonical -> native.canonical
    native4d -> native4d
    native:sequential32 -> native:sequential32
    native.canonical:sequential32 -> native.canonical:sequential32
    native-sequential -> native:sequential32
    native4d:sequential32 -> error: variant "native4d:sequential32": no configuration "sequential32"
    aten:sequential32 -> error: variant "aten:sequential32": no configuration "sequential32"
    native:fast -> error: variant "native:fast": no configuration "fast"
    wasm -> error: unknown variant "wasm" |}]
