open Cram_json
open Cram_program

let unsupported () =
  let n =
    node ~stack:"L__self__,,M;L__self__blk,blk,B" "bogus_operator.default"
      [ ("self", t "x") ]
      "y"
  in
  let meta = update "strides" (J.list [ i 4; i 1 ]) (tm [ 1; 4 ]) in
  let tv = [ ("x", meta); ("y", meta) ] in
  save "operator" (program [ n ] tv);
  let specs =
    [
      obj [ ("user_input", obj [ ("arg", t "x") ]) ];
      obj [ ("token", obj [ ("arg", obj [ ("name", J.string "tok") ]) ]) ];
    ]
  in
  save "input"
    (program ~specs
       [ update "target" (J.string "torch.ops.aten.relu.default") n ]
       tv)

let malformed () =
  let p = read "operator.json" in
  let gm = at "graph_module" p in
  save "malformed"
    (update "graph_module"
       (update "graph" (update "nodes" (J.list []) (at "graph" gm)) gm)
       p)

let malformed_rows () =
  let flat = tm [ 1; 4 ] in
  let relu = node "relu.default" [ ("self", t "x") ] "y" in
  let both = [ ("x", flat); ("y", flat) ] in
  let w name ?(inputs = [ "x" ]) ?(outputs = [ t "y" ]) ns tv =
    save ("m-" ^ name) (program ~inputs ~outputs ns tv)
  in
  w "missing-arg" [ update "inputs" (J.list []) relu ] both;
  w "wrong-kind" [ update "inputs" (J.list [ arg "self" (i 3) ]) relu ] both;
  w "node-output" [ update "outputs" (J.list [ i 1 ]) relu ] both;
  w "graph-output" ~outputs:[ i 1 ] [ relu ] both;
  w "no-metadata" [ relu ] [ ("y", flat) ];
  w "negative" [ relu ] [ ("x", tm [ -1 ]); ("y", flat) ];
  w "symbolic" [ relu ]
    [
      ( "x",
        tm_sizes
          [
            scalar "as_expr"
              (obj [ ("expr_str", J.string "s0"); ("hint", J.null ()) ]);
          ] );
      ("y", flat);
    ];
  w "rank-seven" [ relu ] [ ("x", tm (List.init 7 (fun _ -> 1))); ("y", flat) ];
  w "axis" [ node "mean.dim" [ ("self", t "x"); ("dim", ints [ 9 ]) ] "y" ] both;
  w "alpha"
    [
      node "add.Tensor"
        [ ("self", t "x"); ("other", t "x"); ("alpha", i 2) ]
        "y";
    ]
    both;
  w "memory-format"
    [
      node "clone.default"
        [
          ("self", t "x"); ("memory_format", scalar "as_memory_format" (J.int 2));
        ]
        "y";
    ]
    both;
  let conv ?(padding = [ 0; 0 ]) ?(groups = 1) stride =
    node "convolution.default"
      [
        ("input", t "x");
        ("weight", t "w");
        ("bias", scalar "as_none" (J.bool true));
        ("stride", ints stride);
        ("padding", ints padding);
        ("dilation", ints [ 1; 1 ]);
        ("transposed", b false);
        ("output_padding", ints [ 0; 0 ]);
        ("groups", i groups);
      ]
      "y"
  in
  let img = tm [ 1; 3; 8; 8 ] and out = tm [ 1; 2; 8; 8 ] in
  let tv = [ ("x", img); ("w", tm [ 2; 3; 1; 1 ]); ("y", out) ] in
  w "arity" ~inputs:[ "x"; "w" ] [ conv [ 1; 1; 1 ] ] tv;
  w "conv-rank" ~inputs:[ "x"; "w" ]
    [ conv [ 1; 1 ] ]
    [ ("x", img); ("w", tm [ 2; 3 ]); ("y", out) ];
  w "config-pos" ~inputs:[ "x"; "w" ] [ conv ~groups:0 [ 1; 1 ] ] tv;
  w "config-neg" ~inputs:[ "x"; "w" ] [ conv ~padding:[ -1; -1 ] [ 1; 1 ] ] tv;
  w "zero-dim" [ relu ] [ ("x", tm [ 0; 4 ]); ("y", flat) ];
  let names =
    scalar "as_tensors"
      (J.list
         (List.map (fun n -> obj [ ("name", J.string n) ]) [ "y"; "z"; "w" ]))
  in
  w "output-arity"
    [ node ~outputs:[ names ] "unbind.int" [ ("self", t "x") ] "y" ]
    [ ("x", tm [ 2; 4 ]); ("y", flat) ]

let unbind () =
  let case name shape dim count out_shape =
    let names = List.init count (fun i -> Printf.sprintf "u%d" i) in
    let ins =
      [ ("self", t "x") ]
      @ match dim with None -> [] | Some n -> [ ("dim", i n) ]
    in
    let output =
      scalar "as_tensors"
        (J.list (List.map (fun n -> obj [ ("name", J.string n) ]) names))
    in
    save name
      (program
         ~outputs:[ t "u0" ]
         [ node ~outputs:[ output ] "unbind.int" ins "u0" ]
         (("x", tm shape) :: List.map (fun n -> (n, tm out_shape)) names))
  in
  case "u-dim-absent" [ 2; 3; 4 ] None 2 [ 3; 4 ];
  case "u-dim-pos" [ 2; 3; 4 ] (Some 1) 3 [ 2; 4 ];
  case "u-dim-neg" [ 2; 3; 4 ] (Some (-1)) 4 [ 2; 3 ];
  case "u-vit-rank5" [ 3; 1; 3; 101; 32 ] None 3 [ 1; 3; 101; 32 ];
  case "u-count-mismatch" [ 2; 3; 4 ] (Some 0) 3 [ 3; 4 ];
  case "u-over-limit" [ 4096; 2 ] (Some 0) 4096 [ 2 ];
  save "u-graph-list"
    (program
       ~outputs:
         [ scalar "as_tensors" (J.list [ obj [ ("name", J.string "y") ] ]) ]
       [ node "relu.default" [ ("self", t "x") ] "y" ]
       [ ("x", tm [ 2; 3 ]); ("y", tm [ 2; 3 ]) ])
