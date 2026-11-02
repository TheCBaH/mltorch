(* A case: one graph, the options its invocations are planned under, and
   whether the rates are fitted to it or only checked against it. *)

module T = Machine_model_test.Model_test
module B = Machine_model_test.Attention_binding_test
module AT = Machine_model_test.Attention_layer_test
module LF = Native_test.Attention_layer_fixtures
module L = Attention_layer.Attention_layer
module F = Native_test.Graph_fixtures
module S = Machine_model.Sdpa_select
module Layer_select = Machine_model.Layer_select
module A = Attention_block

(* Alphabetical. *)
module Role = struct
  type t = Check | Fit
end

(* Where every tensor lives. Alphabetical. *)
module Home = struct
  type t = Default | Ram
end

type t = {
  name : string;
  role : Role.t;
  graph : Graph_ir.graph;
  target : Ssa_ir.Ssa_target.t;
  contractions : bool;
  bf16_pairs : bool;
  attention :
    (Graph_ir.Node_id.t -> Machine_model.Attention_site.t option) option;
  layer : (Graph_ir.Node_id.t -> Machine_model.Layer_site.t option) option;
  oracle : T.oracle;
  input : salt:int -> int -> Tensor_sig.t -> Tensor.packed;
  home : Home.t;
}

let pipeline c =
  Ssa_backends.Pipeline.Planned
    { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target = c.target }

let base ?(role = Role.Fit) ?(target = Ssa_ir.Ssa_target.zve32f128)
    ?(contractions = false) ?(bf16_pairs = false) ?(home = Home.Default) name
    graph =
  {
    name;
    role;
    graph;
    target;
    contractions;
    bf16_pairs;
    attention = None;
    layer = None;
    oracle = T.Fp32;
    input = (fun ~salt i sg -> T.in_format sg (T.values ~salt:(salt + i) sg));
    home;
  }

let scalar = Ssa_ir.Ssa_target.scalar

let unary ~h ~w name f =
  F.build name
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 h w) () in
      f x)

let binary ~h ~w name f =
  F.build name
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 h w) () in
      let* y = input ~shape:(F.s 1 1 1 1 h w) () in
      f x y)

let positive ~salt i sg =
  let t = T.values ~salt:(salt + i) sg in
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      1. +. Float.abs (Tensor.read t c))

let norm ~h ~w name f =
  F.build name
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 h w) () in
      let* weight = input ~shape:(F.s 1 1 1 1 1 w) () in
      let* bias = input ~shape:(F.s 1 1 1 1 1 w) () in
      f x weight bias)

let eps = Json_util.f32_to_f32 1e-5

let attention ?(role = Role.Fit) ?(block_keys = 32) ?home name g =
  let node = B.sdpa_node g in
  let grant =
    Some { S.Grant.block_keys; precision = A.Precision.Bf16_paired }
  in
  match S.choose ~grant B.engine g node with
  | Error r -> Fmt.failwith "%s: not selected: %a" name S.Reason.pp r
  | Ok site ->
      {
        (base ~role ?home name g) with
        attention =
          Some
            (fun id ->
              if Graph_ir.Node_id.equal id node then Some site else None);
        oracle = B.attention_oracle site;
        input =
          (fun ~salt i sg ->
            T.in_format sg
              (if i = 3 then
                 Tensor.materialize sg.Tensor_sig.shape (fun c ->
                     let n = (Vec6.offset sg.Tensor_sig.shape c :> int) in
                     if (n + salt) mod 7 = 0 then neg_infinity
                     else float_of_int (n mod 5) /. 4.)
               else T.values ~salt:(salt + (3 * i)) sg));
      }

let layer_base =
  {
    (LF.all_flags LF.small) with
    LF.Config.tokens = 16;
    width = 64;
    heads = 2;
    batch = 1;
    bf16_weights = true;
  }

let layer ?(role = Role.Check) ?(home = Home.Default) name c =
  let g = LF.node c in
  let node = AT.layer_node g in
  let grant =
    Some { S.Grant.block_keys = 16; precision = A.Precision.Bf16_paired }
  in
  match Layer_select.choose ~grant B.engine g node with
  | Error r -> Fmt.failwith "%s: not selected: %a" name Layer_select.Reason.pp r
  | Ok site ->
      let values = LF.inputs g in
      {
        (base ~role ~home name g) with
        layer =
          Some
            (fun id ->
              if Graph_ir.Node_id.equal id node then Some site else None);
        oracle = AT.layer_oracle site;
        input =
          (fun ~salt:_ _ (sg : Tensor_sig.t) ->
            T.in_format sg (List.assoc sg.Tensor_sig.id values));
      }

let bare =
  {
    layer_base with
    LF.Config.q_bias = false;
    k_bias = false;
    v_bias = false;
    o_bias = false;
    layer_scale = false;
  }

(* bfloat16 Q, K, V over [heads] on H *)
let sdpa_heads ~heads ~tq ~tk ~d =
  let bf16 = Payload.Fmt Payload.BF16 in
  F.build "sdpa_bf16_heads"
    Graph_builder.(
      let* q = input ~shape:(F.s 1 1 1 heads tq d) ~fmt:bf16 () in
      let* k = input ~shape:(F.s 1 1 1 heads tk d) ~fmt:bf16 () in
      let* v = input ~shape:(F.s 1 1 1 heads tk d) ~fmt:bf16 () in
      sdpa
        { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
        ~query:q ~key:k ~value:v ())

(* [L, E] split into [h] heads of [d]: reshape, then heads before tokens, as a
   layer's projections are split *)
let head_split ~l ~h ~d =
  F.build "head_split"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 l (h * d)) () in
      let* x = reshape { Reshape.Reshape.shape = F.s 1 1 1 l h d } x in
      permute Axis.[ (N, N); (T, T); (D, D); (H, W); (W, H); (C, C) ] x)

(* The fit set: every class, kernel and memory region, each at two or three
   sizes so a per-invocation cost is told from a per-element one. *)
let fit =
  let g name graph = base name graph in
  let s name graph = base ~target:scalar name graph in
  let pos name graph = { (g name graph) with input = positive } in
  let spos name graph = { (s name graph) with input = positive } in
  let add ~h ~w = binary ~h ~w "add" (fun x y -> Graph_builder.add x y) in
  let mul ~h ~w = binary ~h ~w "mul" (fun x y -> Graph_builder.mul x y) in
  let div ~h ~w = binary ~h ~w "div" (fun x y -> Graph_builder.div x y) in
  let transpose ~n =
    unary ~h:n ~w:n "transpose" (fun x ->
        Graph_builder.permute
          Axis.[ (N, N); (T, T); (D, D); (H, H); (W, C); (C, W) ]
          x)
  in
  let softmax ~h ~w =
    unary ~h ~w "softmax" (fun x ->
        Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x)
  in
  let sum ~h ~w axis =
    unary ~h ~w "sum" (fun x ->
        Graph_builder.sum { Reduce.Sum.dims = [ axis ]; keepdim = true } x)
  in
  let ram = function Some Home.Ram -> ", in RAM" | _ -> "" in
  let lin ?home ~m ~n ~k () =
    base ?home ~contractions:true
      (Fmt.str "linear %dx%d k%d%s" m n k (ram home))
      (T.linear_fixture ~m ~n ~k ~bias:false ())
  in
  let lin16 ?home ~m ~n ~k () =
    base ?home ~contractions:true ~bf16_pairs:true
      (Fmt.str "linear bf16 %dx%d k%d%s" m n k (ram home))
      (T.linear_bf16_fixture ~m ~n ~k ~bias:false ())
  in
  let att ?block_keys ?home ~tq ~tk ~d () =
    attention ?block_keys ?home
      (Fmt.str "sdpa bf16 %dx%d d%d%s%s" tq tk d
         (match block_keys with Some b -> Fmt.str " bk%d" b | None -> "")
         (ram home))
      (T.sdpa_bf16_fixture ~tq ~tk ~d ())
  in
  let unaries =
    List.concat_map
      (fun (name, f, positive_only) ->
        List.concat_map
          (fun (h, w) ->
            let graph = unary ~h ~w name f in
            let label = Fmt.str "%s %dx%d" name h w in
            if positive_only then
              [ pos label graph; spos (label ^ ", scalar") graph ]
            else [ g label graph; s (label ^ ", scalar") graph ])
          [ (16, 16); (32, 32) ])
      [
        ("cos", (fun x -> Graph_builder.cos x), false);
        ("exp", (fun x -> Graph_builder.exp x), false);
        ("log", (fun x -> Graph_builder.log x), true);
        ("sin", (fun x -> Graph_builder.sin x), false);
        ("sqrt", (fun x -> Graph_builder.sqrt x), true);
      ]
  in
  let binaries =
    List.concat_map
      (fun (name, f) ->
        List.concat_map
          (fun (h, w) ->
            let graph = f ~h ~w in
            let label = Fmt.str "%s %dx%d" name h w in
            [ g label graph; s (label ^ ", scalar") graph ])
          [ (16, 64); (32, 128) ])
      [
        ("add", add);
        ("div", div);
        ("mul", mul);
        (* other ratios of loads to operations, so the classes separate *)
        ( "x*2",
          fun ~h ~w ->
            unary ~h ~w "scale" (fun x -> Graph_builder.mul_scalar 2. x) );
        ( "x*x+x",
          fun ~h ~w ->
            unary ~h ~w "square_add" (fun x ->
                Graph_builder.(
                  let* y = mul x x in
                  add y x)) );
        ( "x+y+z",
          fun ~h ~w ->
            F.build "add3"
              Graph_builder.(
                let* x = input ~shape:(F.s 1 1 1 1 h w) () in
                let* y = input ~shape:(F.s 1 1 1 1 h w) () in
                let* z = input ~shape:(F.s 1 1 1 1 h w) () in
                let* s = add x y in
                add s z) );
      ]
  in
  unaries @ binaries
  @ [
      base ~home:Home.Ram "add 16x64, in RAM" (add ~h:16 ~w:64);
      base ~home:Home.Ram "add 32x128, in RAM" (add ~h:32 ~w:128);
      (* a vector operation that takes longer than its loads, from RAM: the
         loads and the operation overlap *)
      base ~home:Home.Ram "div 16x64, in RAM" (div ~h:16 ~w:64);
      base ~home:Home.Ram "div 32x128, in RAM" (div ~h:32 ~w:128);
      base ~home:Home.Ram "exp 32x32, in RAM"
        (unary ~h:32 ~w:32 "exp" (fun x -> Graph_builder.exp x));
      base ~home:Home.Ram ~target:scalar "add 16x64, scalar, in RAM"
        (add ~h:16 ~w:64);
      g "softmax over C 16x16" (softmax ~h:16 ~w:16);
      g "softmax over C 32x32" (softmax ~h:32 ~w:32);
      g "sum over C 16x64" (sum ~h:16 ~w:64 Axis.C);
      g "sum over C 32x128" (sum ~h:32 ~w:128 Axis.C);
      g "sum over W 16x64" (sum ~h:16 ~w:64 Axis.W);
      g "transpose 32x32" (transpose ~n:32);
      g "transpose 64x64" (transpose ~n:64);
      base ~home:Home.Ram "transpose 32x32, in RAM" (transpose ~n:32);
      g "head split 16x64, 2 heads" (head_split ~l:16 ~h:2 ~d:32);
      g "head split 32x128, 4 heads" (head_split ~l:32 ~h:4 ~d:32);
      g "head split 49x64, 2 heads" (head_split ~l:49 ~h:2 ~d:32);
      lin ~m:9 ~n:32 ~k:64 ();
      lin ~m:16 ~n:64 ~k:64 ();
      lin ~m:64 ~n:64 ~k:64 ();
      lin ~m:16 ~n:128 ~k:32 ();
      lin ~m:32 ~n:32 ~k:128 ();
      lin ~m:48 ~n:80 ~k:32 ();
      lin16 ~m:9 ~n:32 ~k:64 ();
      lin16 ~m:16 ~n:64 ~k:64 ();
      lin16 ~m:64 ~n:64 ~k:64 ();
      lin16 ~m:32 ~n:128 ~k:32 ();
      lin16 ~m:48 ~n:48 ~k:48 ();
      att ~tq:32 ~tk:32 ~d:32 ();
      att ~tq:64 ~tk:64 ~d:64 ();
      att ~tq:16 ~tk:48 ~d:32 ();
      att ~block_keys:16 ~tq:16 ~tk:16 ~d:32 ();
      att ~block_keys:16 ~tq:32 ~tk:64 ~d:64 ();
      att ~tq:64 ~tk:32 ~d:32 ();
      lin ~home:Home.Ram ~m:9 ~n:32 ~k:64 ();
      lin ~home:Home.Ram ~m:32 ~n:32 ~k:64 ();
      (* B past the panels' share of the DTCM: the plain function *)
      lin16 ~m:16 ~n:256 ~k:64 ();
      lin16 ~m:32 ~n:128 ~k:128 ();
      lin16 ~m:49 ~n:512 ~k:64 ();
      lin16 ~home:Home.Ram ~m:16 ~n:256 ~k:64 ();
      lin16 ~home:Home.Ram ~m:16 ~n:64 ~k:64 ();
      lin16 ~home:Home.Ram ~m:64 ~n:64 ~k:64 ();
      att ~home:Home.Ram ~tq:32 ~tk:32 ~d:32 ();
      att ~home:Home.Ram ~block_keys:16 ~tq:16 ~tk:16 ~d:32 ();
      att ~home:Home.Ram ~tq:64 ~tk:64 ~d:64 ();
      attention "sdpa bf16 9x29 d64 masked"
        (T.sdpa_bf16_fixture ~tq:9 ~tk:29 ~d:64 ~mask:(9, 29) ());
    ]

(* The check set: shapes and compositions the rates were not fitted to. *)
let check =
  let c name graph = base ~role:Role.Check name graph in
  [
    c "exp 64x64" (unary ~h:64 ~w:64 "exp" Graph_builder.exp);
    c "softmax over C 64x64"
      (unary ~h:64 ~w:64 "softmax" (fun x ->
           Graph_builder.softmax { Reduce.Softmax.axis = Axis.C } x));
    c "softmax over W 16x16"
      (unary ~h:16 ~w:16 "softmax" (fun x ->
           Graph_builder.softmax { Reduce.Softmax.axis = Axis.W } x));
    c "layer norm over C 16x64"
      (norm ~h:16 ~w:64 "layer_norm" (fun x weight bias ->
           Graph_builder.layer_norm
             { Norm.LayerNorm.dims = [ Axis.C ]; eps }
             ~x ~weight ~bias ()));
    c "rms norm over C 16x64"
      (norm ~h:16 ~w:64 "rms_norm" (fun x weight _ ->
           Graph_builder.rms_norm
             { Norm.RmsNorm.dims = [ Axis.C ]; eps }
             ~x ~weight ()));
    base ~role:Role.Check ~contractions:true "linear 32x96 k48"
      (T.linear_fixture ~m:32 ~n:96 ~k:48 ~bias:false ());
    attention ~role:Role.Check "sdpa bf16 41x61 d32"
      (T.sdpa_bf16_fixture ~tq:41 ~tk:61 ~d:32 ());
    layer "layer, bare" bare;
    layer "layer, all flags" layer_base;
    layer "layer, pre layer norm, rope"
      (LF.with_norm L.Norm.Kind.Layer L.Placement.Pre
         { layer_base with LF.Config.rope = LF.Rope_form.Shared });
    layer "layer, post rms"
      (LF.with_norm L.Norm.Kind.Rms L.Placement.Post layer_base);
    {
      (base ~role:Role.Check ~contractions:true "layer decomposed, bare"
         (LF.decomposed { bare with LF.Config.bf16_weights = false }))
      with
      input = (fun ~salt i sg -> T.values ~salt:(salt + i) sg);
    };
  ]

(* Measured by other suites (too long for the calibration run): checked
   against the cycles their goldens record. *)
let elsewhere =
  [
    ( layer ~home:Home.Ram "fastvit 49 x 512, 16 x 32"
        {
          layer_base with
          LF.Config.tokens = 49;
          width = 512;
          heads = 16;
          q_bias = false;
          k_bias = false;
          v_bias = false;
        },
      "../machine_rivet_riscv32_coral_fastvit/fastvit_test.ml" );
  ]

let all = fit @ check
