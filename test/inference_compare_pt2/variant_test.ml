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
      "native:arena,scheduled";
      "native:sequential32,arena=4096,scheduled";
      "native.canonical:separate";
      "native:scheduled";
      "native:arena,shared";
      "native:arena=-1";
      "native4d:arena";
      "native:symbolic";
      "native.canonical:symbolic";
      "native4d:symbolic";
      "aten:symbolic";
      "native:symbolic,arena";
      "native:symbolic,sequential32";
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
    native:arena,scheduled -> native:arena,scheduled
    native:sequential32,arena=4096,scheduled -> native:sequential32,arena=4096,scheduled
    native.canonical:separate -> native.canonical:separate
    native:scheduled -> error: variant "native:scheduled": scheduled needs release, arena, separate or shared
    native:arena,shared -> error: variant "native:arena,shared": more than one storage
    native:arena=-1 -> error: variant "native:arena=-1": no configuration "arena=-1"
    native4d:arena -> error: variant "native4d:arena": no configuration "arena"
    native:symbolic -> native:symbolic
    native.canonical:symbolic -> native.canonical:symbolic
    native4d:symbolic -> native4d:symbolic
    aten:symbolic -> error: variant "aten:symbolic": no configuration "symbolic"
    native:symbolic,arena -> error: variant "native:symbolic,arena": symbolic takes no other word
    native:symbolic,sequential32 -> error: variant "native:symbolic,sequential32": symbolic takes no other word
    wasm -> error: unknown variant "wasm" |}]
