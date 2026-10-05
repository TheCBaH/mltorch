open Ssa_bridge
open Ssa_fixtures
open Ssa_ir
module B = Ssa_builder
module Fx = Ssa_ir_test.Ssa_fixtures

(* The numerical policies on kernels big enough for the plans to differ: the
   sweeps' walks are small, so they rarely schedule a sum or leave binary64. *)

let neon = Ssa_target.neon128
let relaxed = Ssa_numerics.Simd_fp32_relaxed
let ordered = Ssa_numerics.Simd_fp32_ordered
let r32 = Ssa_const.round_f32

(* The schedule of {!Ssa_vector_sum}, written out from its definition, over
   binary32 terms and binary32 adds. *)
let scheduled_sum ~lanes ~parts ~seed (terms : float array) =
  let n = Array.length terms in
  let full = n / lanes in
  let rounds = full / parts and extra = full mod parts in
  let tail = n - (full * lanes) in
  let acc = Array.make_matrix parts lanes 0. in
  for round = 0 to rounds - 1 do
    for j = 0 to parts - 1 do
      for k = 0 to lanes - 1 do
        acc.(j).(k) <-
          r32 (acc.(j).(k) +. terms.((((round * parts) + j) * lanes) + k))
      done
    done
  done;
  for e = 0 to extra - 1 do
    for k = 0 to lanes - 1 do
      acc.(e).(k) <-
        r32 (acc.(e).(k) +. terms.((((rounds * parts) + e) * lanes) + k))
    done
  done;
  let rec tree = function
    | [] -> invalid_arg "tree"
    | [ x ] -> x
    | xs ->
        let rec pairs = function
          | a :: b :: rest -> r32 (a +. b) :: pairs rest
          | rest -> rest
        in
        tree (pairs xs)
  in
  let combined =
    List.init lanes (fun k -> tree (List.init parts (fun j -> acc.(j).(k))))
  in
  let horizontal = tree combined in
  let tail_sum = ref 0. in
  for u = 0 to tail - 1 do
    tail_sum := r32 (!tail_sum +. terms.((full * lanes) + u))
  done;
  r32 (seed +. r32 (horizontal +. !tail_sum))

let sequential_sum (terms : float array) =
  Array.fold_left (fun acc x -> r32 (acc +. x)) 0. terms

(* What an unscheduled sum is at the plan's precision: the exact products added
   in order and rounded once at the end (binary64), or each product and each add
   rounded (binary32). *)
let sequential ~(precision : Ssa_numerics.Precision.t) a b =
  match precision with
  | Ssa_numerics.Precision.F32 ->
      sequential_sum (Array.mapi (fun i x -> r32 (x *. b.(i))) a)
  | Ssa_numerics.Precision.F64 ->
      let acc = ref 0. in
      Array.iteri (fun i x -> acc := !acc +. (x *. b.(i))) a;
      r32 !acc

(* A 1 x K by K x 1 product: one output, a dot product of K terms. *)
let dot_plan k = Fusion_plan.default (matmul_kernel ~m:1 ~k ~n:1)

let output plan (resolved : Ssa_plan.t) ~bind =
  match
    Err.payload (Ssa_lower.Ssa_exec.run plan resolved.Ssa_plan.program ~bind)
  with
  | Ok m -> Tensor.read (Tensor_id.Map.find (tid 2) m) Vec6.origin
  | Error e -> Fmt.failwith "%a" Ssa_lower.Ssa_exec.pp_error e

let parts_of (r : Ssa_plan.t) =
  List.filter_map
    (fun (d : Ssa_vector_sum.Decision.t) ->
      match d.Ssa_vector_sum.Decision.outcome with
      | Ssa_vector_sum.Decision.Scheduled { parts } -> Some parts
      | Ssa_vector_sum.Decision.Kept_sequential _ -> None)
    r.Ssa_plan.sums

let%expect_test "a scheduled sum is exactly its definition" =
  List.iter
    (fun k ->
      let a = operand 3 k and b = operand 5 k in
      let bind = matmul_bind ~m:1 ~k ~n:1 ~a ~b in
      let plan = dot_plan k in
      let verdict, resolved =
        Ssa_check.run_planned ~numerics:relaxed ~target:neon plan ~bind
      in
      match resolved with
      | None -> Fmt.pr "K=%d: refused@." k
      | Some r ->
          let terms = Array.init k (fun i -> r32 (a.(i) *. b.(i))) in
          let expected =
            match parts_of r with
            | [ parts ] -> scheduled_sum ~lanes:16 ~parts ~seed:0. terms
            | _ -> sequential ~precision:r.Ssa_plan.precision a b
          in
          Fmt.pr "K=%3d: %s parts=%s; %a; matches the definition: %b@." k
            (Ssa_numerics.Precision.name r.Ssa_plan.precision)
            (String.concat "," (List.map string_of_int (parts_of r)))
            Ssa_check.pp_verdict verdict
            (Core.Float_bits.equal_portable (output plan r ~bind) expected))
    [ 31; 64; 65; 70; 100; 127; 200; 230; 260 ];
  (* more data on the four-part schedules, whose trees differ from a chain *)
  List.iter
    (fun seed ->
      let k = 260 in
      let a = operand seed k and b = operand (seed + 9) k in
      let bind = matmul_bind ~m:1 ~k ~n:1 ~a ~b in
      let plan = dot_plan k in
      match
        snd (Ssa_check.run_planned ~numerics:relaxed ~target:neon plan ~bind)
      with
      | None -> ()
      | Some r ->
          let terms = Array.init k (fun i -> r32 (a.(i) *. b.(i))) in
          let parts = List.hd (parts_of r) in
          Fmt.pr "seed %d, %d parts: %b@." seed parts
            (Core.Float_bits.equal_portable (output plan r ~bind)
               (scheduled_sum ~lanes:16 ~parts ~seed:0. terms)))
    [ 1; 2; 3; 4; 5 ];
  [%expect
    {|
    K= 31: f64 parts=; agree; matches the definition: true
    K= 64: f32 parts=2; agree; matches the definition: true
    K= 65: f32 parts=2; agree; matches the definition: true
    K= 70: f32 parts=2; agree; matches the definition: true
    K=100: f32 parts=3; agree; matches the definition: true
    K=127: f32 parts=3; agree; matches the definition: true
    K=200: f32 parts=4; agree; matches the definition: true
    K=230: f32 parts=4; agree; matches the definition: true
    K=260: f32 parts=4; agree; matches the definition: true
    seed 1, 4 parts: true
    seed 2, 4 parts: true
    seed 3, 4 parts: true
    seed 4, 4 parts: true
    seed 5, 4 parts: true |}]

(* Small integers are exact in binary32 however a sum is ordered, so the
   scheduled sum must equal the sequential one: no term lost, none repeated. *)
let%expect_test "a scheduled sum of integers loses and repeats nothing" =
  List.iter
    (fun k ->
      let a = Array.init k (fun i -> float_of_int ((i mod 7) - 3)) in
      let b = Array.init k (fun i -> float_of_int ((i * 5 mod 11) - 5)) in
      let bind = matmul_bind ~m:1 ~k ~n:1 ~a ~b in
      let plan = dot_plan k in
      let _, resolved =
        Ssa_check.run_planned ~numerics:relaxed ~target:neon plan ~bind
      in
      match resolved with
      | None -> ()
      | Some r ->
          let exact = ref 0. in
          Array.iteri (fun i x -> exact := !exact +. (x *. b.(i))) a;
          Fmt.pr "K=%3d: scheduled=%b exact=%b@." k
            (parts_of r <> [])
            (Float.equal (output plan r ~bind) !exact))
    [ 64; 65; 99; 128; 211 ];
  [%expect
    {|
    K= 64: scheduled=true exact=true
    K= 65: scheduled=true exact=true
    K= 99: scheduled=true exact=true
    K=128: scheduled=true exact=true
    K=211: scheduled=true exact=true |}]

let%expect_test "the ordered policy keeps every sum sequential" =
  List.iter
    (fun k ->
      let a = operand 3 k and b = operand 5 k in
      let bind = matmul_bind ~m:1 ~k ~n:1 ~a ~b in
      let plan = dot_plan k in
      let verdict, resolved =
        Ssa_check.run_planned ~numerics:ordered ~target:neon plan ~bind
      in
      match resolved with
      | None -> ()
      | Some r ->
          Fmt.pr "K=%3d: %s scheduled=%b; %a; the sequential sum: %b@." k
            (Ssa_numerics.Precision.name r.Ssa_plan.precision)
            (parts_of r <> [])
            Ssa_check.pp_verdict verdict
            (Core.Float_bits.equal_portable (output plan r ~bind)
               (sequential ~precision:r.Ssa_plan.precision a b)))
    [ 64; 100 ];
  [%expect
    {|
    K= 64: f64 scheduled=false; agree; the sequential sum: true
    K=100: f64 scheduled=false; agree; the sequential sum: true |}]

(* ---- contraction ----------------------------------------------------------- *)

let bufs =
  [
    Fx.buffer 0 ~h:1L ~w:80L Ssa_format.F32 Ssa_buffer.Input;
    Fx.buffer 1 ~h:1L ~w:80L Ssa_format.F32 Ssa_buffer.Output;
  ]

let idx bld n = B.index bld (Int64.of_int n)

let load_at bld i =
  B.load_f64 bld (Fx.buf 0) ~decode:Ssa_op.Decode.F32_to_f64
    (Fx.at bld ~h:(idx bld 0) ~w:i)

(* out[w] = body x, over [trips] iterations. *)
let kernel ~trips body =
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ~buffers:bufs (fun bld ->
         let B.Nil =
           B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld trips) ~init:B.Nil
             (fun bld w B.Nil ->
               B.store_f64 bld (Fx.buf 1) ~encode:Ssa_op.Encode.F32_round
                 (Fx.at bld ~h:(idx bld 0) ~w)
                 (body bld (load_at bld w));
               B.Nil)
         in
         ()))

let mul bld a b = B.f64_binary bld Expr.Value.Mul a b
let add bld a b = B.f64_binary bld Expr.Value.Add a b

(* out[w] = x * x + c *)
let multiply_add =
  kernel ~trips:64 (fun bld x -> add bld (mul bld x x) (B.f64 bld 0.1))

let cells ?fused p input =
  let out = Array.make 80 0. in
  let memory =
    Fx.memory [ (0, Fx.floats (Array.copy input)); (1, Fx.floats out) ]
  in
  (match Err.payload (Ssa_interp.run ?fused p ~memory) with
  | Ok () -> ()
  | Error e -> Fmt.failwith "%a" Ssa_interp.pp_error e);
  out

let%expect_test "contraction needs the permission and a fused operation" =
  let input =
    Array.init 80 (fun i -> r32 (1. +. (float_of_int i *. 0.0123457)))
  in
  let alias = Ssa_effects.Distinct_buffers in
  let plan numerics target =
    Ssa_plan.resolve ~target ~alias ~numerics multiply_add
  in
  let describe name (r : Ssa_plan.t) =
    Fmt.pr "%-34s precision=%s vector-loops=%d fused=%d@." name
      (Ssa_numerics.Precision.name r.Ssa_plan.precision)
      (List.length r.Ssa_plan.vectorized)
      r.Ssa_plan.contracted
  in
  describe "ordered, fused target" (plan ordered neon);
  describe "relaxed, fused target" (plan relaxed neon);
  describe "relaxed, relaxed madd (vectors)"
    (plan relaxed Ssa_target.wasm128_relaxed);
  describe "relaxed, no fused operation" (plan relaxed Ssa_target.wasm128);
  describe "reference" (plan Ssa_numerics.Reference_f64 neon);
  let r = plan relaxed neon in
  let got = cells r.Ssa_plan.program input in
  let expected =
    Array.map
      (fun x -> Ssa_numerics.fma32 (r32 x) (r32 x) (r32 0.1) |> r32)
      input
  in
  let head a = Array.sub a 0 64 in
  Fmt.pr "fused result is fmaf: %b@."
    (Array.for_all2 Core.Float_bits.equal_portable (head got) (head expected));
  let unfused = cells ~fused:false r.Ssa_plan.program input in
  let differs =
    Array.exists2
      (fun a b -> not (Core.Float_bits.equal_portable a b))
      (head got) (head unfused)
  in
  Fmt.pr "the unfused reading differs somewhere: %b@." differs;
  let ordered_result = cells (plan ordered neon).Ssa_plan.program input in
  Fmt.pr "ordered result is the separate multiply and add: %b@."
    (Array.for_all2 Core.Float_bits.equal_portable (head ordered_result)
       (head unfused));
  [%expect
    {|
    ordered, fused target              precision=f32 vector-loops=1 fused=0
    relaxed, fused target              precision=f32 vector-loops=1 fused=1
    relaxed, relaxed madd (vectors)    precision=f32 vector-loops=1 fused=1
    relaxed, no fused operation        precision=f32 vector-loops=1 fused=0
    reference                          precision=f64 vector-loops=1 fused=0
    fused result is fmaf: true
    the unfused reading differs somewhere: true
    ordered result is the separate multiply and add: true |}]

(* A multiplication on each side of the add: the right one is fused, as the
   Loop planner's contraction takes it, and the left one stays a product. *)
let%expect_test
    "contraction takes the right-hand product, and the tail only on a scalar \
     target" =
  let input =
    Array.init 80 (fun i -> r32 (1. +. (float_of_int i *. 0.0123457)))
  in
  let alias = Ssa_effects.Distinct_buffers in
  let both =
    kernel ~trips:64 (fun bld x ->
        add bld (mul bld x x) (mul bld x (B.f64 bld 1.7)))
  in
  let r = Ssa_plan.resolve ~target:neon ~alias ~numerics:relaxed both in
  let got = cells r.Ssa_plan.program input in
  let expected =
    Array.map
      (fun x ->
        let x = r32 x in
        Ssa_numerics.fma32 x (r32 1.7) (r32 (x *. x)))
      input
  in
  Fmt.pr "right-hand product fused: %b@."
    (Array.for_all2 Core.Float_bits.equal_portable (Array.sub got 0 64)
       (Array.sub expected 0 64));
  (* 70 iterations: sixty-four lanes' worth and a scalar tail of six *)
  let with_tail =
    kernel ~trips:70 (fun bld x -> add bld (mul bld x x) (B.f64 bld 0.1))
  in
  let fused target =
    (Ssa_plan.resolve ~target ~alias ~numerics:relaxed with_tail)
      .Ssa_plan.contracted
  in
  Fmt.pr "scalar fused operation: %d; vector-only madd: %d@." (fused neon)
    (fused Ssa_target.wasm128_relaxed);
  [%expect
    {|
    right-hand product fused: true
    scalar fused operation: 2; vector-only madd: 1 |}]
