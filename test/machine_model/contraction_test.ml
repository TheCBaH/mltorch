(* The matrix-product fixtures ({!Model_test.contraction_fixtures}): the
   adversarial inputs that separate the binary32 policies, and the contracted
   policy against its oracle. Every pipeline and route is in
   [contraction_routes_test.ml]. *)

open Graph_ir
open Model_test
module M = Machine_model.Mir_model

(* The adversarial inputs round: binary32 against the binary64 reference
   differs, and most outputs stay finite. *)
let%expect_test "matrix products: adversarial inputs separate the policies" =
  List.iter
    (fun (name, g) ->
      let g' = g () in
      let b = bundle g' in
      let constants =
        List.map
          (fun id -> (id, adversarial ~salt:3 (sig_of g' id)))
          b.Loop_ir.Loop_bundle.constants
      in
      let inputs =
        List.mapi
          (fun i id -> (id, adversarial ~salt:i (sig_of g' id)))
          b.Loop_ir.Loop_bundle.inputs
      in
      let out = List.hd g'.Graph.outputs in
      let f64 =
        Tensor_id.Map.find out (expected Reference_f64 g' b ~constants ~inputs)
      in
      let f32 =
        Tensor_id.Map.find out (expected Fp32 g' b ~constants ~inputs)
      in
      let finite =
        List.length
          (List.filter
             (fun x -> Float.is_finite (Int64.float_of_bits x))
             (bits f32))
      in
      Fmt.pr "%s: %d of %d finite; binary32 oracle %s the binary64 reference@."
        name finite
        (List.length (bits f32))
        (if bits f32 = bits f64 then "equals" else "differs from"))
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64: 195 of 288 finite; binary32 oracle differs from the binary64 reference
    linear 9x32 k64 bias: 195 of 288 finite; binary32 oracle differs from the binary64 reference
    linear 7x29 k63 bias: 154 of 203 finite; binary32 oracle differs from the binary64 reference
    bmm 1x9x32 k64: 196 of 288 finite; binary32 oracle differs from the binary64 reference
    bmm 3x7x29 k63: 456 of 609 finite; binary32 oracle differs from the binary64 reference |}]

let%expect_test "matrix products: multiply-adds" =
  List.iter
    (fun (name, g) -> Fmt.pr "%s: %d@." name (contraction_macs g))
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64: 18432
    linear 9x32 k64 bias: 18432
    linear 7x29 k63 bias: 12789
    bmm 1x9x32 k64: 18432
    bmm 3x7x29 k63: 38367 |}]

(* The contracted oracle is the reference's binary32 sequential dot: the Loop
   interpreter with each single-product sum fused, against [Direct] under
   [Binary32_sequential], two independent evaluators. With the bias added,
   the reference adds it in binary64 and rounds once on store, as the oracle's
   binary32 add does. The unfused oracle differs on adversarial inputs. *)
let%expect_test "matrix products: the contracted oracle is the sequential dot" =
  List.iter
    (fun (name, g) ->
      let g' = g () in
      let b = bundle g' in
      List.iter
        (fun (iname, input) ->
          let constants =
            List.map
              (fun id -> (id, input ~salt:3 (sig_of g' id)))
              b.Loop_ir.Loop_bundle.constants
          in
          let inputs =
            List.mapi
              (fun i id -> (id, input ~salt:i (sig_of g' id)))
              b.Loop_ir.Loop_bundle.inputs
          in
          let out = List.hd g'.Graph.outputs in
          let get o =
            Tensor_id.Map.find out (expected o g' b ~constants ~inputs)
          in
          let direct =
            Direct.with_dot_accumulation Direct.Binary32_sequential (fun () ->
                get Reference_f64)
          in
          let contracted = get Fp32_contracted and ordered = get Fp32 in
          Fmt.pr "%s, %s: contracted %s the sequential reference; ordered %s@."
            name iname
            (if same_portable contracted direct then "equals"
             else "DIFFERS from")
            (if same_portable ordered direct then "equals it" else "differs"))
        [
          ("values", fun ~salt sg -> values ~salt sg);
          ("adversarial", fun ~salt sg -> adversarial ~salt sg);
        ])
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, values: contracted equals the sequential reference; ordered equals it
    linear 9x32 k64, adversarial: contracted equals the sequential reference; ordered differs
    linear 9x32 k64 bias, values: contracted equals the sequential reference; ordered equals it
    linear 9x32 k64 bias, adversarial: contracted equals the sequential reference; ordered differs
    linear 7x29 k63 bias, values: contracted equals the sequential reference; ordered equals it
    linear 7x29 k63 bias, adversarial: contracted equals the sequential reference; ordered differs
    bmm 1x9x32 k64, values: contracted equals the sequential reference; ordered equals it
    bmm 1x9x32 k64, adversarial: contracted equals the sequential reference; ordered differs
    bmm 3x7x29 k63, values: contracted equals the sequential reference; ordered equals it
    bmm 3x7x29 k63, adversarial: contracted equals the sequential reference; ordered differs |}]

(* [Fp32_contracted]: each single-product sum's accumulation fused, on the
   generic interpreter and both targets' scanned programs, scalar and
   vectorized, against the contracted oracle. *)
let%expect_test "matrix products: the contracted policy is its oracle" =
  let contracted target =
    Ssa_backends.Pipeline.Planned
      { numerics = Ssa_ir.Ssa_numerics.Fp32_contracted; target }
  in
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (pname, pipeline) ->
          List.iter
            (fun (rname, route) ->
              check ?route ~pipeline ~oracle:Fp32_contracted ~same:same_portable
                ~input:adversarial_input
                ~constant:(fun sg -> adversarial ~salt:3 sg)
                (Fmt.str "%s, %s, %s" name pname rname)
                g)
            [
              ("generic", None);
              ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
              ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
            ])
        [
          ("scalar", contracted Ssa_ir.Ssa_target.scalar);
          ("neon128", contracted Ssa_ir.Ssa_target.neon128);
        ])
    (contraction_fixtures @ [ List.nth vector_fixtures 3 ]);
  [%expect
    {|
    linear 9x32 k64, scalar, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64, scalar, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 9x32 k64, neon128, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64, neon128, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 9x32 k64 bias, scalar, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, scalar, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 9x32 k64 bias, neon128, generic (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, neon128, aarch64 (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 7x29 k63 bias, scalar, generic (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, scalar, aarch64 (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    linear 7x29 k63 bias, neon128, generic (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, neon128, aarch64 (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 1x9x32 k64, scalar, generic (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, scalar, aarch64 (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 1x9x32 k64, neon128, generic (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, neon128, aarch64 (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 3x7x29 k63, scalar, generic (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, scalar, aarch64 (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    bmm 3x7x29 k63, neon128, generic (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, neon128, aarch64 (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    conv, batch norm, relu to C=32, scalar, generic (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, scalar, aarch64 (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, scalar, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include
    conv, batch norm, relu to C=32, neon128, generic (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, neon128, aarch64 (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, neon128, x86_64: refused: invocation 0 (n0): needs fma, which the program's features do not include |}]

(* What the recognizer finds in each fixture's scalar binary32 form, and why
   it refuses the rest: the matrix products are found with their extents and
   strides (A's rows, B's columns, the batch); every other sum is refused with
   its reason. *)
let%expect_test "contractions found in the fixtures" =
  List.iter
    (fun (name, g) ->
      let b = bundle (g ()) in
      List.iteri
        (fun i inv ->
          match Ssa_backends.program Ssa_backends.Pipeline.Exact inv with
          | Error e -> Fmt.pr "%s %d: %s@." name i e
          | Ok p ->
              List.iter
                (function
                  | Ssa_ir.Ssa_contract.Outcome.Found f ->
                      Fmt.pr "%s %d: %a@." name i Contraction.pp
                        f.Ssa_ir.Ssa_contract.Found.desc
                  | Ssa_ir.Ssa_contract.Outcome.Refused r ->
                      Fmt.pr "%s %d: refused %s@." name i
                        (Ssa_ir.Ssa_contract.Refusal.name r))
                (Ssa_ir.Ssa_contract.census
                   ~numerics:Contraction.Numerics.Ordered
                   (Ssa_ir.Ssa_plan.binary32_scalar p)))
        b.Loop_ir.Loop_bundle.invocations)
    (contraction_fixtures @ vector_fixtures);
  [%expect
    {|
    linear 9x32 k64 0: contract f32 batch=1 m=9 n=32 k=64 a=(0,64,1) b=(0,64,1) c=(288,32,1) seed=zero ordered
    linear 9x32 k64 bias 0: contract f32 batch=1 m=9 n=32 k=64 a=(0,64,1) b=(0,64,1) c=(288,32,1) seed=zero ordered
    linear 7x29 k63 bias 0: contract f32 batch=1 m=7 n=29 k=63 a=(0,63,1) b=(0,63,1) c=(203,29,1) seed=zero ordered
    bmm 1x9x32 k64 0: contract f32 batch=1 m=9 n=32 k=64 a=(0,64,1) b=(0,1,32) c=(288,32,1) seed=zero ordered
    bmm 3x7x29 k63 0: contract f32 batch=3 m=7 n=29 k=63 a=(441,63,1) b=(1827,1,29) c=(203,29,1) seed=zero ordered
    softmax over H, W=32 0: refused not_product
    layer_norm over H, W=32 0: refused not_product
    layer_norm over H, W=32 0: refused not_product
    conv, batch norm, relu to C=32 0: refused bounds
    conv, batch norm, relu to C=32 0: refused not_product
    conv, batch norm, relu to C=32 0: refused not_product |}]

(* ---- the contraction rewrite ---------------------------------------------- *)

(* An invocation's scalar binary32 program run by the SSA interpreter, its
   buffers bound to the graph's tensors through the invocation's edges. *)
let ssa_run ?engine (inv : Loop_ir.Loop_bundle.invocation) ~tensors p =
  let edge =
    List.combine
      (List.map
         (fun (b : Loop_ir.Loop_buffer.t) -> b.Loop_ir.Loop_buffer.id)
         inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers)
      inv.Loop_ir.Loop_bundle.edges
  in
  Err.payload
    (Ssa_lower.Ssa_exec.run ?engine inv.Loop_ir.Loop_bundle.placed p
       ~bind:(fun id ->
         Option.bind (List.assoc_opt id edge) (fun t ->
             List.assoc_opt t tensors)))

let same_outputs a b =
  Tensor_id.Map.equal
    (fun x y ->
      List.for_all2 Core.Float_bits.equal_portable
        (List.map Int64.float_of_bits (bits x))
        (List.map Int64.float_of_bits (bits y)))
    a b

(* The descriptor of each [Contract] in a program, changed by [f]. *)
let map_contracts f (p : Ssa_ir.Ssa_program.t) =
  let rec region (r : Ssa_ir.Ssa_region.t) =
    { r with Ssa_ir.Ssa_region.body = List.map stmt r.Ssa_ir.Ssa_region.body }
  and stmt = function
    | Ssa_ir.Ssa_stmt.Instr
        ({ Ssa_ir.Ssa_instr.op = Ssa_ir.Ssa_op.Contract c; _ } as i) ->
        Ssa_ir.Ssa_stmt.Instr
          {
            i with
            Ssa_ir.Ssa_instr.op =
              Ssa_ir.Ssa_op.Contract { c with desc = f c.desc };
          }
    | Ssa_ir.Ssa_stmt.Instr _ as s -> s
    | Ssa_ir.Ssa_stmt.For f ->
        Ssa_ir.Ssa_stmt.For { f with body = region f.body }
    | Ssa_ir.Ssa_stmt.Ordered_sum o ->
        Ssa_ir.Ssa_stmt.Ordered_sum { o with body = region o.body }
    | Ssa_ir.Ssa_stmt.If i ->
        Ssa_ir.Ssa_stmt.If
          { i with then_ = region i.then_; else_ = region i.else_ }
  in
  { p with Ssa_ir.Ssa_program.entry = region p.Ssa_ir.Ssa_program.entry }

let remake ?(k = fun x -> x) ?(transpose = false) ?numerics (d : Contraction.t)
    =
  let open Contraction in
  let a =
    if transpose then
      { d.a with Operand.outer = d.a.Operand.k; k = d.a.Operand.outer }
    else d.a
  in
  Err.or_raise ~pp_error
    (make ~batch:d.batch ~m:d.m ~n:d.n ~k:(k d.k) ~a ~b:d.b ~c:d.c
       ~format:d.format ~seed:d.seed
       ~numerics:(Option.value numerics ~default:d.numerics))

(* Rewritten, every fixture's program verifies and computes the same bits as
   before under the ordered requirement, on both interpreters; under the
   fused one it is the contracted oracle. Three wrong descriptors are caught:
   A transposed (its row and k strides exchanged: out of bounds or wrong
   cells), K one short, and the fused requirement where the loops were
   ordered. *)
let%expect_test "the contraction rewrite preserves results" =
  List.iter
    (fun (name, g) ->
      let g' = g () in
      let b = bundle g' in
      let constants =
        List.map
          (fun id -> (id, adversarial ~salt:3 (sig_of g' id)))
          b.Loop_ir.Loop_bundle.constants
      and inputs =
        List.mapi
          (fun i id -> (id, adversarial ~salt:i (sig_of g' id)))
          b.Loop_ir.Loop_bundle.inputs
      in
      let tensors = constants @ inputs in
      List.iter
        (fun (inv : Loop_ir.Loop_bundle.invocation) ->
          let p =
            Ssa_ir.Ssa_plan.binary32_scalar
              (Result.get_ok
                 (Ssa_backends.program Ssa_backends.Pipeline.Exact inv))
          in
          let show what q =
            Fmt.str "%s %s" what
              (match (ssa_run inv ~tensors q, ssa_run inv ~tensors p) with
              | Ok x, Ok y -> if same_outputs x y then "bitwise" else "DIFFERS"
              | Error _, _ | _, Error _ -> "refused")
          in
          let q, found =
            Ssa_ir.Ssa_contract.program ~numerics:Contraction.Numerics.Ordered p
          in
          let verified =
            match Err.payload (Ssa_ir.Ssa_verify.check q) with
            | Ok () -> "verifies"
            | Error _ -> "does not verify"
          in
          let cfg =
            match
              ( ssa_run ~engine:Ssa_lower.Ssa_exec.Cfg inv ~tensors q,
                ssa_run inv ~tensors p )
            with
            | Ok x, Ok y -> if same_outputs x y then "bitwise" else "DIFFERS"
            | _ -> "fails"
          in
          let fused, _ =
            Ssa_ir.Ssa_contract.program
              ~numerics:Contraction.Numerics.Fused_sequential p
          in
          let oracle = expected Fp32_contracted g' b ~constants ~inputs in
          let contracted =
            match ssa_run inv ~tensors fused with
            | Ok x ->
                let out = List.hd g'.Graph.outputs in
                if
                  same_portable
                    (Tensor_id.Map.find (Tensor_id.Map.choose x |> fst) x)
                    (Tensor_id.Map.find out oracle)
                then "the contracted oracle"
                else "DIFFERS from the contracted oracle"
            | Error _ -> "fails"
          in
          Fmt.pr
            "%s: %d contraction(s), %s; %s, cfg %s; fused is %s; %s; %s; %s@."
            name (List.length found) verified (show "ordered" q) cfg contracted
            (show "A transposed" (map_contracts (remake ~transpose:true) q))
            (show "K one short" (map_contracts (remake ~k:Int64.pred) q))
            (show "fused"
               (map_contracts
                  (remake ~numerics:Contraction.Numerics.Fused_sequential)
                  q)))
        b.Loop_ir.Loop_bundle.invocations)
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64: 1 contraction(s), verifies; ordered bitwise, cfg bitwise; fused is the contracted oracle; A transposed refused; K one short DIFFERS; fused DIFFERS
    linear 9x32 k64 bias: 1 contraction(s), verifies; ordered bitwise, cfg bitwise; fused is the contracted oracle; A transposed refused; K one short DIFFERS; fused DIFFERS
    linear 7x29 k63 bias: 1 contraction(s), verifies; ordered bitwise, cfg bitwise; fused is the contracted oracle; A transposed refused; K one short DIFFERS; fused DIFFERS
    bmm 1x9x32 k64: 1 contraction(s), verifies; ordered bitwise, cfg bitwise; fused is the contracted oracle; A transposed refused; K one short DIFFERS; fused DIFFERS
    bmm 3x7x29 k63: 1 contraction(s), verifies; ordered bitwise, cfg bitwise; fused is the contracted oracle; A transposed refused; K one short DIFFERS; fused DIFFERS |}]

(* Only the Machine IR route realizes a contraction (as a helper call): the
   C, Wasm and JavaScript emitters each refuse it with a reason. *)
let%expect_test "a contraction is refused where it has no form" =
  let g = List.assoc "linear 9x32 k64" contraction_fixtures in
  let inv = List.hd (bundle (g ())).Loop_ir.Loop_bundle.invocations in
  let q, _ =
    Ssa_ir.Ssa_contract.program ~numerics:Contraction.Numerics.Ordered
      (Ssa_ir.Ssa_plan.binary32_scalar
         (Result.get_ok (Ssa_backends.program Ssa_backends.Pipeline.Exact inv)))
  in
  let says pp = function Ok _ -> "accepted" | Error e -> Fmt.str "%a" pp e in
  Fmt.pr "c: %s@." (says Ssa_c.pp_error (Ssa_c.kernel ~name:"k" q));
  Fmt.pr "wasm: %s@." (says Ssa_wasm.pp_error (Ssa_wasm.lower q));
  Fmt.pr "js: %s@." (says Ssa_js.pp_error (Ssa_js.program q));
  Fmt.pr "machine: %s@."
    (says Machine_lower.Mir_lower.Refusal.pp
       (Err.payload
          (Machine_lower.Mir_lower.program
             ~planning:
               (Some
                  (Machine_ir.Mir_planning.make
                     ~subject:(Machine_lower.Mir_lower.subject q)
                     ~policy:"test" ~schedule:"scalar"
                     ~precision:Machine_ir.Mir_planning.Precision.F32
                     ~lanes:(Machine_ir.Mir_type.Lanes.of_int 1)
                     ~fma:Machine_ir.Mir_planning.Fma.Forbidden ~capabilities:[]))
             q)));
  [%expect
    {|
    c: C has no form for a contraction
    wasm: a contraction has no WebAssembly form
    js: a contraction has no JavaScript form
    machine: accepted |}]

(* With contractions, each matrix product of a binary32 plan is one helper
   call, run by the helper's model: on the generic interpreter and on RV32's
   selected and scanned programs, bitwise against each policy's oracle. *)
let%expect_test "contractions as helper calls on the interpreters" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (pname, numerics, oracle) ->
          List.iter
            (fun (rname, route) ->
              check ?route ~contractions:true
                ~pipeline:
                  (Ssa_backends.Pipeline.Planned
                     { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
                ~oracle ~same:same_portable ~input:adversarial_input
                ~constant:(fun sg -> adversarial ~salt:3 sg)
                ~calls:1
                (Fmt.str "%s, %s, %s" name pname rname)
                g)
            [
              ("generic", None);
              ("riscv32 selected", Some (M.Route.Riscv32 M.Stage.Selected));
              ("riscv32 scanned", Some (M.Route.Riscv32 M.Stage.Scanned));
            ])
        [
          ("ordered", Ssa_ir.Ssa_numerics.Fp32_ordered, Fp32);
          ("contracted", Ssa_ir.Ssa_numerics.Fp32_contracted, Fp32_contracted);
        ])
    contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, ordered, generic (1 invocations): bitwise
    linear 9x32 k64, ordered, riscv32 selected (1 invocations): bitwise
    linear 9x32 k64, ordered, riscv32 scanned (1 invocations): bitwise
    linear 9x32 k64, contracted, generic (1 invocations): bitwise
    linear 9x32 k64, contracted, riscv32 selected (1 invocations): bitwise
    linear 9x32 k64, contracted, riscv32 scanned (1 invocations): bitwise
    linear 9x32 k64 bias, ordered, generic (1 invocations): bitwise
    linear 9x32 k64 bias, ordered, riscv32 selected (1 invocations): bitwise
    linear 9x32 k64 bias, ordered, riscv32 scanned (1 invocations): bitwise
    linear 9x32 k64 bias, contracted, generic (1 invocations): bitwise
    linear 9x32 k64 bias, contracted, riscv32 selected (1 invocations): bitwise
    linear 9x32 k64 bias, contracted, riscv32 scanned (1 invocations): bitwise
    linear 7x29 k63 bias, ordered, generic (1 invocations): bitwise
    linear 7x29 k63 bias, ordered, riscv32 selected (1 invocations): bitwise
    linear 7x29 k63 bias, ordered, riscv32 scanned (1 invocations): bitwise
    linear 7x29 k63 bias, contracted, generic (1 invocations): bitwise
    linear 7x29 k63 bias, contracted, riscv32 selected (1 invocations): bitwise
    linear 7x29 k63 bias, contracted, riscv32 scanned (1 invocations): bitwise
    bmm 1x9x32 k64, ordered, generic (1 invocations): bitwise
    bmm 1x9x32 k64, ordered, riscv32 selected (1 invocations): bitwise
    bmm 1x9x32 k64, ordered, riscv32 scanned (1 invocations): bitwise
    bmm 1x9x32 k64, contracted, generic (1 invocations): bitwise
    bmm 1x9x32 k64, contracted, riscv32 selected (1 invocations): bitwise
    bmm 1x9x32 k64, contracted, riscv32 scanned (1 invocations): bitwise
    bmm 3x7x29 k63, ordered, generic (1 invocations): bitwise
    bmm 3x7x29 k63, ordered, riscv32 selected (1 invocations): bitwise
    bmm 3x7x29 k63, ordered, riscv32 scanned (1 invocations): bitwise
    bmm 3x7x29 k63, contracted, generic (1 invocations): bitwise
    bmm 3x7x29 k63, contracted, riscv32 selected (1 invocations): bitwise
    bmm 3x7x29 k63, contracted, riscv32 scanned (1 invocations): bitwise |}]

(* 1x1 convolutions: an unpadded, unit-stride one is found as an ordered
   product of flattened rows (its window sums hold one product each, so even
   the contracted policy rounds every product) and runs as a helper call,
   bitwise against its oracle; a padded or strided one is refused. *)
let%expect_test "1x1 convolutions as matrix products" =
  List.iter
    (fun (name, g) ->
      let b = bundle (g ()) in
      List.iteri
        (fun i inv ->
          match Ssa_backends.program Ssa_backends.Pipeline.Exact inv with
          | Error e -> Fmt.pr "%s %d: %s@." name i e
          | Ok p ->
              List.iter
                (function
                  | Ssa_ir.Ssa_contract.Outcome.Found f ->
                      Fmt.pr "%s %d: %a@." name i Contraction.pp
                        f.Ssa_ir.Ssa_contract.Found.desc
                  | Ssa_ir.Ssa_contract.Outcome.Refused r ->
                      Fmt.pr "%s %d: refused %s@." name i
                        (Ssa_ir.Ssa_contract.Refusal.name r))
                (Ssa_ir.Ssa_contract.census
                   ~numerics:Contraction.Numerics.Fused_sequential
                   (Ssa_ir.Ssa_plan.binary32_scalar p)))
        b.Loop_ir.Loop_bundle.invocations;
      List.iter
        (fun (rname, route) ->
          check ?route ~contractions:true
            ~pipeline:
              (Ssa_backends.Pipeline.Planned
                 {
                   numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
                   target = Ssa_ir.Ssa_target.zve32f128;
                 })
            ~oracle:Fp32_contracted ~same:same_portable ~input:adversarial_input
            ~constant:(fun sg -> adversarial ~salt:3 sg)
            ~calls:1
            (Fmt.str "%s, %s" name rname)
            g)
        [
          ("generic", None);
          ("riscv32 scanned", Some (M.Route.Riscv32 M.Stage.Scanned));
        ])
    pointwise_fixtures;
  [%expect
    {|
    conv1x1 3x5 c19 to 29 0: contract f32 batch=1 m=15 n=29 k=19 a=(0,19,1) b=(0,19,1) c=(435,29,1) seed=zero ordered
    conv1x1 3x5 c19 to 29, generic (1 invocations): bitwise
    conv1x1 3x5 c19 to 29, riscv32 scanned (1 invocations): bitwise
    conv1x1 3x5 c1 to 4 0: contract f32 batch=1 m=15 n=4 k=1 a=(0,1,1) b=(0,1,1) c=(60,4,1) seed=zero ordered
    conv1x1 3x5 c1 to 4, generic (1 invocations): bitwise
    conv1x1 3x5 c1 to 4, riscv32 scanned (1 invocations): bitwise
    conv1x1 2x2 c8 to 4 pad 1 0: refused bounds
    conv1x1 2x2 c8 to 4 pad 1 0: refused bounds
    conv1x1 2x2 c8 to 4 pad 1 0: refused bounds
    conv1x1 2x2 c8 to 4 pad 1, generic (1 invocations): bitwise
    conv1x1 2x2 c8 to 4 pad 1, riscv32 scanned (1 invocations): bitwise
    conv1x1 4x4 c8 to 4 stride 2 0: refused non_affine
    conv1x1 4x4 c8 to 4 stride 2 0: refused non_affine
    conv1x1 4x4 c8 to 4 stride 2 0: refused non_affine
    conv1x1 4x4 c8 to 4 stride 2, generic (1 invocations): bitwise
    conv1x1 4x4 c8 to 4 stride 2, riscv32 scanned (1 invocations): bitwise |}]

(* Matrix products of bfloat16 operands: found with their format, and run as
   helper calls under both binary32 policies, bitwise against their oracles. *)
let%expect_test "bfloat16 matrix products" =
  List.iter
    (fun (name, g) ->
      let b = bundle (g ()) in
      List.iteri
        (fun i inv ->
          match Ssa_backends.program Ssa_backends.Pipeline.Exact inv with
          | Error e -> Fmt.pr "%s %d: %s@." name i e
          | Ok p ->
              List.iter
                (function
                  | Ssa_ir.Ssa_contract.Outcome.Found f ->
                      Fmt.pr "%s %d: %a@." name i Contraction.pp
                        f.Ssa_ir.Ssa_contract.Found.desc
                  | Ssa_ir.Ssa_contract.Outcome.Refused r ->
                      Fmt.pr "%s %d: refused %s@." name i
                        (Ssa_ir.Ssa_contract.Refusal.name r))
                (Ssa_ir.Ssa_contract.census
                   ~numerics:Contraction.Numerics.Ordered
                   (Ssa_ir.Ssa_plan.binary32_scalar p)))
        b.Loop_ir.Loop_bundle.invocations;
      List.iter
        (fun (pname, numerics, oracle) ->
          List.iter
            (fun (rname, route) ->
              check ?route ~contractions:true
                ~pipeline:
                  (Ssa_backends.Pipeline.Planned
                     { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
                ~oracle ~same:same_portable
                ~input:(fun ~salt id sg ->
                  in_format sg (adversarial_input ~salt id sg))
                ~constant:(fun sg -> in_format sg (adversarial ~salt:3 sg))
                ~calls:1
                (Fmt.str "%s, %s, %s" name pname rname)
                g)
            [
              ("generic", None);
              ("riscv32 scanned", Some (M.Route.Riscv32 M.Stage.Scanned));
            ])
        [
          ("ordered", Ssa_ir.Ssa_numerics.Fp32_ordered, Fp32);
          ("contracted", Ssa_ir.Ssa_numerics.Fp32_contracted, Fp32_contracted);
        ])
    bf16_fixtures;
  [%expect
    {|
    bmm bf16 3x7x29 k63 0: contract bf16 batch=3 m=7 n=29 k=63 a=(441,63,1) b=(1827,1,29) c=(203,29,1) seed=zero ordered
    bmm bf16 3x7x29 k63, ordered, generic (1 invocations): bitwise
    bmm bf16 3x7x29 k63, ordered, riscv32 scanned (1 invocations): bitwise
    bmm bf16 3x7x29 k63, contracted, generic (1 invocations): bitwise
    bmm bf16 3x7x29 k63, contracted, riscv32 scanned (1 invocations): bitwise
    linear bf16 9x32 k64 0: contract bf16 batch=1 m=9 n=32 k=64 a=(0,64,1) b=(0,64,1) c=(288,32,1) seed=zero ordered
    linear bf16 9x32 k64, ordered, generic (1 invocations): bitwise
    linear bf16 9x32 k64, ordered, riscv32 scanned (1 invocations): bitwise
    linear bf16 9x32 k64, contracted, generic (1 invocations): bitwise
    linear bf16 9x32 k64, contracted, riscv32 scanned (1 invocations): bitwise
    linear bf16 7x29 k63 bias 0: contract bf16 batch=1 m=7 n=29 k=63 a=(0,63,1) b=(0,63,1) c=(203,29,1) seed=zero ordered
    linear bf16 7x29 k63 bias, ordered, generic (1 invocations): bitwise
    linear bf16 7x29 k63 bias, ordered, riscv32 scanned (1 invocations): bitwise
    linear bf16 7x29 k63 bias, contracted, generic (1 invocations): bitwise
    linear bf16 7x29 k63 bias, contracted, riscv32 scanned (1 invocations): bitwise |}]

(* Prepacked weights: a Linear's weight, B with strided columns, read as a
   packed copy the context builds once; a Bmm's B is an input and stays.
   Bitwise under both policies, call after call (the packed copy is built
   once and read by every call). *)
let%expect_test "prepacked weights" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (pname, numerics, oracle) ->
          List.iter
            (fun (rname, route) ->
              check ?route ~contractions:true ~prepack:true
                ~pipeline:
                  (Ssa_backends.Pipeline.Planned
                     { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
                ~oracle ~same:same_portable
                ~input:(fun ~salt id sg ->
                  in_format sg (adversarial_input ~salt id sg))
                ~constant:(fun sg -> in_format sg (adversarial ~salt:3 sg))
                ~calls:2
                (Fmt.str "%s, %s, %s" name pname rname)
                g)
            [
              ("generic", None);
              ("riscv32 scanned", Some (M.Route.Riscv32 M.Stage.Scanned));
            ])
        [
          ("ordered", Ssa_ir.Ssa_numerics.Fp32_ordered, Fp32);
          ("contracted", Ssa_ir.Ssa_numerics.Fp32_contracted, Fp32_contracted);
        ])
    (List.filter
       (fun (n, _) ->
         List.mem n
           [
             "linear 7x29 k63 bias";
             "bmm 3x7x29 k63";
             "linear bf16 7x29 k63 bias";
             "conv1x1 3x5 c19 to 29";
           ])
       (contraction_fixtures @ bf16_fixtures @ pointwise_fixtures));
  [%expect
    {|
    linear 7x29 k63 bias, ordered, generic (1 invocations, 1 packed): bitwise, bitwise
    linear 7x29 k63 bias, ordered, riscv32 scanned (1 invocations, 1 packed): bitwise, bitwise
    linear 7x29 k63 bias, contracted, generic (1 invocations, 1 packed): bitwise, bitwise
    linear 7x29 k63 bias, contracted, riscv32 scanned (1 invocations, 1 packed): bitwise, bitwise
    bmm 3x7x29 k63, ordered, generic (1 invocations, 0 packed): bitwise, bitwise
    bmm 3x7x29 k63, ordered, riscv32 scanned (1 invocations, 0 packed): bitwise, bitwise
    bmm 3x7x29 k63, contracted, generic (1 invocations, 0 packed): bitwise, bitwise
    bmm 3x7x29 k63, contracted, riscv32 scanned (1 invocations, 0 packed): bitwise, bitwise
    linear bf16 7x29 k63 bias, ordered, generic (1 invocations, 1 packed): bitwise, bitwise
    linear bf16 7x29 k63 bias, ordered, riscv32 scanned (1 invocations, 1 packed): bitwise, bitwise
    linear bf16 7x29 k63 bias, contracted, generic (1 invocations, 1 packed): bitwise, bitwise
    linear bf16 7x29 k63 bias, contracted, riscv32 scanned (1 invocations, 1 packed): bitwise, bitwise
    conv1x1 3x5 c19 to 29, ordered, generic (1 invocations, 1 packed): bitwise, bitwise
    conv1x1 3x5 c19 to 29, ordered, riscv32 scanned (1 invocations, 1 packed): bitwise, bitwise
    conv1x1 3x5 c19 to 29, contracted, generic (1 invocations, 1 packed): bitwise, bitwise
    conv1x1 3x5 c19 to 29, contracted, riscv32 scanned (1 invocations, 1 packed): bitwise, bitwise |}]
