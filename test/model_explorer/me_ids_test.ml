(* [Me_ids]: the injective encoder, the layer vocabulary, and where each of the
   two ceilings applies (.ai/model_explorer_design.md).

   The property that matters is INJECTIVITY, and it is the one a golden alone
   cannot state: every "renders sensibly" case below would pass under an
   encoder that maps two labels to one id. So the encoder is driven over a
   corpus chosen for near-collisions and the distinctness of the results is
   asserted, not eyeballed. *)

let pp_id ppf r =
  Core.Pretty.err_result ~ok:(Fmt.fmt "%S") ~error:Me_ids.pp_error ppf r

let show r = Format.printf "%a@." pp_id r
let show_labelled label r = Format.printf "%-28s %a@." label pp_id r
let limits = Me_limits.Limits.untrusted

(* --- the encoder --- *)

let%expect_test "every character the renderer reads is escaped" =
  List.iter
    (fun s -> show_labelled (Printf.sprintf "%S" s) (Me_ids.component s))
    [
      "plain";
      "a/b";
      "a;b";
      "a\\b";
      "a#b";
      "a:b";
      "a%b";
      "layer1/conv/weight";
      "naïve — 世界";
    ];
  [%expect
    {|
    "plain"                      "plain"
    "a/b"                        "a%2Fb"
    "a;b"                        "a%3Bb"
    "a\\b"                       "a%5Cb"
    "a#b"                        "a%23b"
    "a:b"                        "a%3Ab"
    "a%b"                        "a%25b"
    "layer1/conv/weight"         "layer1%2Fconv%2Fweight"
    "na\195\175ve \226\128\148 \228\184\150\231\149\140" "na\195\175ve \226\128\148 \228\184\150\231\149\140" |}]

let%expect_test "the encoding is injective over near-collisions" =
  (* Exactly the corpus a two-pass encoder gets wrong. If [%] were escaped
     AFTER the others, ["a/b"] would become ["a%2Fb"] and the literal
     ["a%2Fb"] would too, so these two rows would collide while every
     individual rendering still looked correct. *)
  let corpus =
    [
      "a/b";
      "a%2Fb";
      "a%2fb";
      "a%252Fb";
      "a;b";
      "a%3Bb";
      "a:b";
      "a%3Ab";
      "a#b";
      "a%23b";
      "a\\b";
      "a%5Cb";
      "a%b";
      "a%25b";
      "";
      "%";
      "%%";
    ]
  in
  let encoded =
    List.map
      (fun s ->
        (s, Err.or_raise ~pp_error:Me_ids.pp_error (Me_ids.component s)))
      corpus
  in
  List.iter (fun (s, e) -> Format.printf "%-10S -> %S@." s e) encoded;
  let outputs = List.map snd encoded in
  let distinct = List.sort_uniq String.compare outputs in
  Printf.printf "%d inputs, %d distinct outputs, injective: %b\n"
    (List.length outputs) (List.length distinct)
    (List.length outputs = List.length distinct);
  [%expect
    {|
    "a/b"      -> "a%2Fb"
    "a%2Fb"    -> "a%252Fb"
    "a%2fb"    -> "a%252fb"
    "a%252Fb"  -> "a%25252Fb"
    "a;b"      -> "a%3Bb"
    "a%3Bb"    -> "a%253Bb"
    "a:b"      -> "a%3Ab"
    "a%3Ab"    -> "a%253Ab"
    "a#b"      -> "a%23b"
    "a%23b"    -> "a%2523b"
    "a\\b"     -> "a%5Cb"
    "a%5Cb"    -> "a%255Cb"
    "a%b"      -> "a%25b"
    "a%25b"    -> "a%2525b"
    ""         -> ""
    "%"        -> "%25"
    "%%"       -> "%25%25"
    17 inputs, 17 distinct outputs, injective: true |}]

let%expect_test "control bytes are refused, not encoded" =
  (* They would survive percent-encoding into an id the renderer displays
     verbatim, and there is no correct rendering of them. *)
  List.iter
    (fun s -> show_labelled (Printf.sprintf "%S" s) (Me_ids.component s))
    [ "a\tb"; "a\nb"; "a\x00b"; "a\x7fb"; "a\x1fb"; "a b" ];
  [%expect
    {|
    "a\tb"                       control byte 0x09 in label
    "a\nb"                       control byte 0x0A in label
    "a\000b"                     control byte 0x00 in label
    "a\127b"                     control byte 0x7F in label
    "a\031b"                     control byte 0x1F in label
    "a b"                        "a b" |}]

(* --- the layer vocabulary --- *)

let%expect_test "the layer vocabulary is closed and round-trips" =
  List.iter
    (fun l ->
      let s = Me_ids.Layer.to_string l in
      match Me_ids.Layer.of_string s with
      | Some l' when l' = l -> print_string (s ^ " ")
      | _ -> Printf.printf "\nBROKEN %s\n" s)
    Me_ids.Layer.all;
  Printf.printf "\n%d layers\n" (List.length Me_ids.Layer.all);
  [%expect {|
    pt2 native native4d symbolic kernel
    5 layers |}]

(* --- assembled ids --- *)

let%expect_test "the structural forms" =
  let path = Pt2_native_graph.Graph_path.root in
  let nested =
    Pt2_native_graph.Graph_path.child
      (Pt2_native_graph.Graph_path.child path 3)
      7
  in
  Format.printf "pt2_graph root   %S@." (Me_ids.pt2_graph path);
  Format.printf "pt2_graph nested %S@." (Me_ids.pt2_graph nested);
  Format.printf "pt2_node         %S@." (Me_ids.pt2_node nested 12);
  Format.printf "graph            %S@." (Me_ids.graph Me_ids.Layer.Native 7);
  Format.printf "graph wide       %S@." (Me_ids.graph Me_ids.Layer.Native 1234);
  Format.printf "flow_state       %S@."
    (Me_ids.flow_state Me_ids.Layer.Native4d 0);
  Format.printf "flow_transition  %S@."
    (Me_ids.flow_transition Me_ids.Layer.Kernel 2);
  Format.printf "op_node          %S@."
    (Me_ids.op_node (Graph_ir.Node_id.of_int 41));
  Format.printf "boundary in      %S@."
    (Me_ids.boundary `In (Graph_ir.Tensor_id.of_int 5));
  Format.printf "boundary const   %S@."
    (Me_ids.boundary `Const (Graph_ir.Tensor_id.of_int 6));
  Format.printf "boundary out     %S@."
    (Me_ids.boundary `Out (Graph_ir.Tensor_id.of_int 7));
  [%expect
    {|
    pt2_graph root   "pt2/root"
    pt2_graph nested "pt2/root/3/7"
    pt2_node         "root/3/7#12"
    graph            "g/native/007"
    graph wide       "g/native/1234"
    flow_state       "s/native4d/000"
    flow_transition  "t/kernel/002"
    op_node          "n41"
    boundary in      "in:t5"
    boundary const   "const:t6"
    boundary out     "out:t7" |}]

let%expect_test "a collection id encodes its dynamic half and nothing else" =
  (* The [mltorch:] prefix carries a colon, which is one of the six characters
     [component] escapes. That it survives here is the whole point of applying
     the encoder to components rather than to assembled ids. *)
  show (Me_ids.collection ~limits "resnet18");
  show (Me_ids.collection ~limits "my/model:v2");
  [%expect {|
    "mltorch:resnet18"
    "mltorch:my%2Fmodel%3Av2" |}]

let%expect_test "namespace components, labelled and not" =
  show (Me_ids.namespace_component ~limits 4);
  show (Me_ids.namespace_component ~limits ~label:"features" 4);
  show (Me_ids.namespace_component ~limits ~label:"features/0" 4);
  [%expect {|
    "g4"
    "features#g4"
    "features%2F0#g4" |}]

let%expect_test "a detail id does not re-encode the ids it is derived from" =
  (* Re-encoding would not be idempotent: the [%] of an existing escape would
     escape again, and the result would no longer name the graph it comes
     from. *)
  let parent_graph = Me_ids.graph Me_ids.Layer.Native 2 in
  let parent_node = Me_ids.op_node (Graph_ir.Node_id.of_int 9) in
  show (Me_ids.detail ~limits ~parent_graph ~parent_node 3);
  [%expect {| "expr/g/native/002/n9/t3" |}]

(* --- the ceilings, and which side of the encoding each is on --- *)

let%expect_test "the raw label is bounded before encoding, the id after" =
  (* Encoding can expand a component threefold, so the two ceilings cannot be
     the same check: a label inside [max_label_bytes] can still produce an id
     past [max_id_bytes], and that is the case below. *)
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_label_bytes:12 ~max_id_bytes:24 limits)
  in
  show_labelled "raw over the label bound"
    (Me_ids.collection ~limits:tight "abcdefghijklm");
  show_labelled "raw inside, id inside"
    (Me_ids.collection ~limits:tight "abcdefghijkl");
  show_labelled "raw inside, id over"
    (Me_ids.collection ~limits:tight "////////////");
  [%expect
    {|
    raw over the label bound     label of 13 bytes is over the ceiling
    raw inside, id inside        "mltorch:abcdefghijkl"
    raw inside, id over          encoded id of 44 bytes is over the ceiling |}]

let%expect_test "a namespace component meets its own ceiling, not max_id_bytes"
    =
  (* Two different fields, because the renderer splits a namespace on ['/'] and
     works over each piece: a component that fits the whole-id ceiling can
     still be one the splitter must carry. *)
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_namespace_component_bytes:8
         ~max_id_bytes:4096 limits)
  in
  show (Me_ids.namespace_component ~limits:tight ~label:"ab" 4);
  show (Me_ids.namespace_component ~limits:tight ~label:"abcdefgh" 4);
  [%expect {|
    "ab#g4"
    encoded id of 11 bytes is over the ceiling |}]
