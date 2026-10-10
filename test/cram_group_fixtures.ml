open Cram_json
open Cram_program

let group2 () =
  let nodes =
    [
      node ~stack:"L__self__,,M;L__self__conv,conv,C" "conv2d.default"
        [
          ("input", t "x");
          ("weight", t "w1");
          ("bias", t "b1");
          ("stride", ints [ 2; 1 ]);
          ("padding", ints [ 1; 0 ]);
          ("dilation", ints [ 1; 2 ]);
          ("groups", i 2);
        ]
        "y1";
      node ~stack:"L__self__,,M;L__self__same,same,C" "conv2d.padding"
        [
          ("input", t "y1");
          ("weight", t "w2");
          ("bias", scalar "as_none" (J.bool true));
          ("stride", ints [ 1; 1 ]);
          ("padding", s "same");
          ("dilation", ints [ 1; 1 ]);
          ("groups", i 1);
        ]
        "y2";
      node ~stack:"L__self__,,M;L__self__pool,pool,P" "max_pool2d.default"
        [
          ("self", t "y2");
          ("kernel_size", ints [ 3; 2 ]);
          ("stride", ints [ 2; 1 ]);
          ("padding", ints [ 1; 0 ]);
        ]
        "y3";
      node ~stack:"L__self__,,M;L__self__norm,norm,N" "rms_norm.default"
        [
          ("input", t "y3");
          ("normalized_shape", ints [ 2; 5 ]);
          ("weight", t "w3");
          ("eps", f 1e-5);
        ]
        "y4";
      node ~stack:"L__self__,,M;L__self__fc,fc,L" "linear.default"
        [ ("input", t "y4"); ("weight", t "w4"); ("bias", t "b4") ]
        "y";
    ]
  in
  simple
    ~params:(named [ "w1"; "b1"; "w2"; "w3"; "w4"; "b4" ])
    "group2" nodes
    [
      ("x", [ 1; 4; 8; 8 ]);
      ("w1", [ 8; 2; 3; 2 ]);
      ("b1", [ 8 ]);
      ("y1", [ 1; 8; 4; 6 ]);
      ("w2", [ 8; 8; 3; 3 ]);
      ("y2", [ 1; 8; 4; 6 ]);
      ("y3", [ 1; 8; 2; 5 ]);
      ("w3", [ 2; 5 ]);
      ("y4", [ 1; 8; 2; 5 ]);
      ("w4", [ 3; 5 ]);
      ("b4", [ 3 ]);
      ("y", [ 1; 8; 2; 3 ]);
    ]

let group3 () =
  simple
    ~params:[ ("other", "other") ]
    "group3"
    [
      node "sub.Tensor" [ ("self", t "x"); ("other", t "other") ] "y1";
      node "sub.Tensor" [ ("self", t "y1"); ("other", i 3) ] "y2";
      node "_unsafe_view.default"
        [ ("self", t "y2"); ("size", ints [ -1; 32 ]) ]
        "y";
    ]
    [
      ("x", [ 1; 4; 8; 8 ]);
      ("other", [ 8 ]);
      ("y1", [ 1; 4; 8; 8 ]);
      ("y2", [ 1; 4; 8; 8 ]);
      ("y", [ 8; 32 ]);
    ];
  let tr name d0 d1 shape =
    simple name
      [
        node "transpose.int"
          [ ("self", t "x"); ("dim0", i d0); ("dim1", i d1) ]
          "y";
      ]
      [ ("x", [ 1; 3; 4; 5 ]); ("y", shape) ]
  in
  tr "group3-transpose" 1 2 [ 1; 4; 3; 5 ];
  tr "group3-transpose-refused" 0 1 [ 3; 1; 4; 5 ];
  simple "group3-malformed"
    [
      node "_unsafe_view.default"
        [ ("self", t "x"); ("size", ints [ -1; -1 ]) ]
        "y";
    ]
    [ ("x", [ 1; 4; 8; 8 ]); ("y", [ 1; 4; 8; 8 ]) ]

let group5 () =
  simple "group5"
    [
      node ~stack:"L__self__,,M;L__self__act1,act1,S" "silu.default"
        [ ("self", t "x") ]
        "y1";
      node ~stack:"L__self__,,M;L__self__act2,act2,H" "hardsigmoid.default"
        [ ("self", t "y1") ]
        "y2";
      node ~stack:"L__self__,,M;L__self__act3,act3,W" "hardswish.default"
        [ ("self", t "y2") ]
        "y";
    ]
    [
      ("x", [ 1; 4; 4; 4 ]);
      ("y1", [ 1; 4; 4; 4 ]);
      ("y2", [ 1; 4; 4; 4 ]);
      ("y", [ 1; 4; 4; 4 ]);
    ]

let group6 () =
  simple "group6"
    [
      node ~stack:"L__self__,,M;L__self__pad,pad,P" "pad.default"
        [
          ("self", t "x");
          ("pad", ints [ 1; 2; -1; 0 ]);
          ("mode", s "constant");
          ("value", f 0.5);
        ]
        "y1";
      node ~stack:"L__self__,,M;L__self__sel,sel,S" "slice.Tensor"
        [
          ("self", t "y1");
          ("dim", i (-1));
          ("start", scalar "as_sym_int" (i 1));
          ("end", i 99);
          ("step", i 2);
        ]
        "y";
    ]
    [ ("x", [ 1; 4; 4; 4 ]); ("y1", [ 1; 4; 3; 7 ]); ("y", [ 1; 4; 3; 3 ]) ]

let outside6 () =
  simple "outside"
    [ node "slice.Tensor" [ ("self", t "x"); ("dim", i 0); ("end", i 2) ] "y" ]
    [ ("x", [ 3; 1; 2; 2; 2 ]); ("y", [ 2; 1; 2; 2; 2 ]) ]

let empty_pad () =
  simple "empty"
    [
      node "pad.default" [ ("self", t "x"); ("pad", ints [ 0; 0; -1; -2 ]) ] "y";
    ]
    [ ("x", [ 1; 4; 2; 4 ]); ("y", [ 1; 4; 2; 4 ]) ]

let empty_slice () =
  simple "empty-slice"
    [
      node "slice.Tensor"
        [ ("self", t "x"); ("dim", i 3); ("start", i 2); ("end", i 2) ]
        "y";
    ]
    [ ("x", [ 1; 4; 2; 4 ]); ("y", [ 1; 4; 2; 4 ]) ]

let group7 () =
  simple
    ~params:[ ("w", "ln.weight"); ("b", "ln.bias") ]
    "group7"
    [
      node ~stack:"L__self__,,M;L__self__ln1,ln1,LN" "layer_norm.default"
        [
          ("input", t "x");
          ("normalized_shape", ints [ 8 ]);
          ("weight", t "w");
          ("bias", t "b");
          ("eps", f 1e-5);
          ("cudnn_enable", b true);
        ]
        "y1";
      node ~stack:"L__self__,,M;L__self__ln2,ln2,LN"
        ~outputs:[ t "y"; t "mean"; t "rstd" ]
        "native_layer_norm.default"
        [
          ("input", t "y1");
          ("normalized_shape", ints [ 8 ]);
          ("weight", t "w");
          ("bias", t "b");
          ("eps", f 1e-6);
        ]
        "y";
    ]
    [
      ("x", [ 1; 4; 8 ]);
      ("w", [ 8 ]);
      ("b", [ 8 ]);
      ("y1", [ 1; 4; 8 ]);
      ("y", [ 1; 4; 8 ]);
    ]

let outside7 () =
  simple "outside"
    [
      node "layer_norm.default"
        [
          ("input", t "x");
          ("normalized_shape", ints [ 2; 3; 4; 5 ]);
          ("eps", f 1e-5);
        ]
        "y";
    ]
    [ ("x", [ 2; 3; 4; 5 ]); ("y", [ 2; 3; 4; 5 ]) ]

let group8 () =
  simple
    ~params:[ ("k", "attn.key"); ("v", "attn.value"); ("m", "attn.mask") ]
    "group8"
    [
      node ~stack:"L__self__,,M;L__self__attn1,attn1,SDPA"
        "scaled_dot_product_attention.default"
        [ ("query", t "x"); ("key", t "k"); ("value", t "v") ]
        "y1";
      node ~stack:"L__self__,,M;L__self__attn2,attn2,SDPA"
        "scaled_dot_product_attention.default"
        [
          ("query", t "y1");
          ("key", t "k");
          ("value", t "v");
          ("attn_mask", t "m");
          ("dropout_p", f 0.);
          ("is_causal", b false);
          ("scale", f 0.1);
        ]
        "y";
    ]
    [
      ("x", [ 1; 1; 2; 4 ]);
      ("k", [ 1; 1; 3; 4 ]);
      ("v", [ 1; 1; 3; 4 ]);
      ("m", [ 2; 3 ]);
      ("y1", [ 1; 1; 2; 4 ]);
      ("y", [ 1; 1; 2; 4 ]);
    ]
