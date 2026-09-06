(* Scratch prototype for LSTM's per-layer/direction scan arithmetic (project
   step 13 / lstm-plan.md §5), isolated from the graph/importer plumbing so the
   new territory -- a packed [2*K]-wide scan doing real weight-matrix loads,
   index arithmetic (the shared divmod helper) and gate nonlinearities -- can
   be verified against a plain OCaml recurrence before any Native op
   registration exists. Not wired into [Graph_ir]; superseded once
   [lib/native/ops/lstm.ml] lands and its own tests cover this ground through
   the graph boundary. *)

open Expr

let k = 2 (* hidden_size *)
let isz = 2 (* input width *)
let seq = 3 (* sequence length *)

(* Layouts: weight_ih [4K,I], weight_hh [4K,K], bias_ih/bias_hh [4K], h0/c0
   [K], input [L,I] -- N holds the packed row, C the column, matching
   [Linear]'s [Out,1,1,1,1,In] convention. *)
let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let seq_shape ~len ~cols = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:len ~w:1 ~c:cols

let tensor_of_array shape arr =
  Tensor.materialize shape (fun coord -> arr.((Vec6.offset shape coord :> int)))

let ids = Hashtbl.create 8
let next_id = ref 0
let f32 = Payload.Fmt Payload.F32

let fresh_source name arr shape =
  let id = Tensor_id.of_int !next_id in
  incr next_id;
  Hashtbl.replace ids (Tensor_id.to_int id) (tensor_of_array shape arr);
  Tensor_sig.create ~id ~name ~shape ~fmt:f32 ()

let env =
  Expr_bridge.env ~binding:(fun id ->
      Hashtbl.find_opt ids (Tensor_id.to_int id))

(* ATen gate order i,f,g,o (lstm-plan.md §5): row block [r*K, (r+1)*K) of the
   packed [4K] weight/bias belongs to gate [r]. *)
let gate_i = 0
let gate_f = 1
let gate_g = 2
let gate_o = 3

let unsafe_floor_div_pos x d =
  match Index.floor_div_pos x d with Ok v -> v | Error _ -> assert false

(* The shared divmod helper (lstm-implementation-plan.md "Stage 2+"): [x mod
   k], encoded from [Add]/[Scale]/[Floor_div_pos] and converted back with
   [Clamp_low] -- sound because [x >= 0] and [k > 0] make the remainder
   provably in [0, k). *)
let mod_k x =
  let q = unsafe_floor_div_pos x k in
  Index.clamp_low (Index.add x (Index.scale (-k) q))

let row_index ~gate k_idx =
  Index.clamp_low (Index.add (Index.const (gate * k)) (Index.of_position k_idx))

let mat_coord ~row ~col =
  Coord.make ~n:row ~t:Index.zero ~d:Index.zero ~h:Index.zero ~w:Index.zero
    ~c:col

let vec_coord row =
  Coord.make ~n:row ~t:Index.zero ~d:Index.zero ~h:Index.zero ~w:Index.zero
    ~c:Index.zero

let seq_coord ~row ~col =
  Coord.make ~n:Index.zero ~t:Index.zero ~d:Index.zero ~h:row ~w:Index.zero
    ~c:col

(* Sum over [0,extent) of [f j], one Region reduction node -- the AST stays
   the same size whatever [extent] is at runtime. *)
let sum_over extent f =
  let open Expr.Builder.Syntax in
  Expr.Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero
    ~hi:(Index.const extent) (fun j ->
      let* v = f j in
      Expr.Builder.return v)

(* One gate's pre-activation: sum_j W_hh[gate,k,j]*prev_h[j]
   + sum_j W_ih[gate,k,j]*layer_input(j) [+ b_hh[gate,k] + b_ih[gate,k]]. *)
let gate_sum ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~previous_at ~layer_input
    ~gate k_idx =
  let open Expr.Builder.Syntax in
  let row = row_index ~gate k_idx in
  let* hh =
    sum_over k (fun j ->
        Expr.Builder.return
          (Value.mul (previous_at j)
             (Region_context.load weight_hh (mat_coord ~row ~col:j))))
  in
  let* ih =
    sum_over isz (fun j ->
        Expr.Builder.return
          (Value.mul (layer_input j)
             (Region_context.load weight_ih (mat_coord ~row ~col:j))))
  in
  let base = Value.add hh ih in
  let base =
    match bias_hh with
    | None -> base
    | Some bias_hh ->
        Value.add base (Region_context.load bias_hh (vec_coord row))
  in
  let base =
    match bias_ih with
    | None -> base
    | Some bias_ih ->
        Value.add base (Region_context.load bias_ih (vec_coord row))
  in
  Expr.Builder.return base

let neg x = Value.sub (Value.const 0.) x

(* [x] is embedded only once: with no expression-level sharing yet (the scan
   design record defers that), any formula referencing its argument twice or
   more duplicates that argument's WHOLE subtree at every reference -- and a
   sign/abs-based stable tanh (guard, negate-and-select, reuse) does that 2-3
   times PER CALL, which compounds badly once tanh is nested (once for the
   [g] gate, again for [next_c]) and reused at the final h/c select. *)
let sigmoid x =
  Value.div (Value.const 1.) (Value.add (Value.const 1.) (Value.exp (neg x)))

(* [tanh(x) = 2*sigmoid(2x) - 1] -- algebraically the stable form
   lstm-plan.md §5 asks for ([exp(-2x)] either overflows to [+inf] or
   underflows to [0], both of which [1/(1+_)] handles exactly, never the
   [inf/inf] NaN a direct [(e^2x-1)/(e^2x+1)] quotient produces for large
   positive [x]), reached through the identity instead of an explicit sign
   branch so [x] is embedded once, not three times. *)
let tanh_v x =
  Value.sub
    (Value.mul (Value.const 2.) (sigmoid (Value.mul (Value.const 2.) x)))
    (Value.const 1.)

let one_step ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~layer_input ~step ~lane
    ~previous_at =
  let open Expr.Builder.Syntax in
  let k_idx = mod_k (Index.of_position lane) in
  let layer_input j = layer_input ~step ~col:j in
  let* a_i =
    gate_sum ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~previous_at ~layer_input
      ~gate:gate_i k_idx
  in
  let* a_f =
    gate_sum ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~previous_at ~layer_input
      ~gate:gate_f k_idx
  in
  let* a_g =
    gate_sum ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~previous_at ~layer_input
      ~gate:gate_g k_idx
  in
  let* a_o =
    gate_sum ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~previous_at ~layer_input
      ~gate:gate_o k_idx
  in
  let i = sigmoid a_i
  and f = sigmoid a_f
  and g = tanh_v a_g
  and o = sigmoid a_o in
  let previous_c =
    previous_at
      (Index.clamp_low (Index.add (Index.const k) (Index.of_position k_idx)))
  in
  let next_c = Value.add (Value.mul f previous_c) (Value.mul i g) in
  let next_h = Value.mul o (tanh_v next_c) in
  let lane_val = Value.value_of_index (Index.of_position lane) in
  Expr.Builder.return
    (Value.select
       (Bool.value_lt lane_val (Value.const (float_of_int k)))
       next_h next_c)

let limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:1024 ~max_updates:100_000L)

let build ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~h0 ~c0 ~input =
  let layer_input ~step ~col =
    Region_context.load input (seq_coord ~row:step ~col)
  in
  let init ~lane =
    let k_idx = mod_k (Index.of_position lane) in
    let is_h =
      Bool.value_lt
        (Value.value_of_index (Index.of_position lane))
        (Value.const (float_of_int k))
    in
    let h_val = Region_context.load h0 (vec_coord k_idx) in
    let c_val = Region_context.load c0 (vec_coord k_idx) in
    Expr.Builder.return (Value.select is_h h_val c_val)
  in
  let update ~step ~lane ~previous_at =
    one_step ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~layer_input ~step ~lane
      ~previous_at
  in
  Region_program.Builder.run
    (Region_program.Builder.scan ~limits ~width:(2 * k) ~steps:seq ~init ~update
       (fun scan_read ->
         Region_program.Builder.finish ~max_size:4096 ~max_depth:128
           ~partition:Region_partition.singleton
           ~output:
             (scan_read ~row:(Index.output Axis.H) ~lane:(Index.output Axis.W))))

let output_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:(seq + 1) ~w:(2 * k) ~c:1

let program ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~h0 ~c0 ~input =
  Err.or_raise ~pp_error:Region_program.pp_error
    (build ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~h0 ~c0 ~input)

(* A plain OCaml recurrence, independent of the Region/Expr machinery, as the
   cross-check oracle. *)
let reference ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~h0 ~c0 ~input =
  let sigmoid x = 1. /. (1. +. exp (-.x)) in
  let get2 arr ~rows:_ ~cols row col = arr.((row * cols) + col) in
  let h = Array.copy h0 and c = Array.copy c0 in
  let trace = Array.make ((seq + 1) * 2 * k) 0. in
  for j = 0 to k - 1 do
    trace.(j) <- h.(j);
    trace.(k + j) <- c.(j)
  done;
  for s = 0 to seq - 1 do
    let gate r j =
      let row = (r * k) + j in
      let hh = ref 0. and ih = ref 0. in
      for jj = 0 to k - 1 do
        hh := !hh +. (get2 weight_hh ~rows:(4 * k) ~cols:k row jj *. h.(jj))
      done;
      for jj = 0 to isz - 1 do
        ih :=
          !ih
          +. get2 weight_ih ~rows:(4 * k) ~cols:isz row jj
             *. input.((s * isz) + jj)
      done;
      !hh +. !ih +. bias_hh.(row) +. bias_ih.(row)
    in
    for j = 0 to k - 1 do
      let i = sigmoid (gate gate_i j) in
      let f = sigmoid (gate gate_f j) in
      let g = tanh (gate gate_g j) in
      let o = sigmoid (gate gate_o j) in
      let next_c = (f *. c.(j)) +. (i *. g) in
      let next_h = o *. tanh next_c in
      trace.(((s + 1) * 2 * k) + j) <- next_h;
      trace.(((s + 1) * 2 * k) + k + j) <- next_c
    done;
    for j = 0 to k - 1 do
      h.(j) <- trace.(((s + 1) * 2 * k) + j);
      c.(j) <- trace.(((s + 1) * 2 * k) + k + j)
    done
  done;
  trace

let%expect_test "scan-based LSTM step agrees with a plain OCaml reference" =
  let weight_hh =
    [|
      0.1;
      -0.2;
      0.05;
      0.3;
      -0.1;
      0.2;
      0.15;
      -0.05;
      0.2;
      0.1;
      -0.3;
      0.1;
      0.05;
      0.2;
      -0.1;
      0.3;
    |]
  in
  let weight_ih =
    [|
      0.2;
      -0.1;
      0.1;
      0.3;
      -0.2;
      0.05;
      0.1;
      -0.3;
      0.05;
      0.2;
      -0.1;
      0.1;
      0.3;
      -0.05;
      0.2;
      -0.2;
    |]
  in
  let bias_hh = [| 0.1; -0.1; 0.2; 0.05; -0.2; 0.1; 0.05; -0.05 |] in
  let bias_ih = [| 0.05; 0.1; -0.1; 0.2; 0.1; -0.05; -0.1; 0.05 |] in
  let h0 = [| 0.3; -0.4 |] in
  let c0 = [| 0.5; 0.1 |] in
  let input = [| 1.0; -0.5; 0.2; 0.7; -0.3; 0.4 |] in
  let weight_hh_t =
    fresh_source "weight_hh" weight_hh (mat_shape ~rows:(4 * k) ~cols:k)
  in
  let weight_ih_t =
    fresh_source "weight_ih" weight_ih (mat_shape ~rows:(4 * k) ~cols:isz)
  in
  let bias_hh_t = fresh_source "bias_hh" bias_hh (vec_shape ~n:(4 * k)) in
  let bias_ih_t = fresh_source "bias_ih" bias_ih (vec_shape ~n:(4 * k)) in
  let h0_t = fresh_source "h0" h0 (vec_shape ~n:k) in
  let c0_t = fresh_source "c0" c0 (vec_shape ~n:k) in
  let input_t = fresh_source "input" input (seq_shape ~len:seq ~cols:isz) in
  let prog =
    program ~weight_hh:weight_hh_t ~weight_ih:weight_ih_t
      ~bias_hh:(Some bias_hh_t) ~bias_ih:(Some bias_ih_t) ~h0:h0_t ~c0:c0_t
      ~input:input_t
  in
  let lowered =
    Err.or_raise ~pp_error:Region_program.pp_error
      (Region_execution.lower_region ~max_size:4096 ~max_depth:128
         ~max_local_slots:8192 ~scan_limits:limits ~output_shape prog)
  in
  let produced =
    Err.or_raise ~pp_error:Region_eval.pp_error
      (Region_execution.materialize lowered ~env)
  in
  let expected =
    reference ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~h0 ~c0 ~input
  in
  let max_diff = ref 0. in
  for row = 0 to seq do
    for lane = 0 to (2 * k) - 1 do
      let p =
        Tensor.read produced (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:row ~w:lane ~c:0)
      in
      let e = expected.((row * 2 * k) + lane) in
      max_diff := Float.max !max_diff (Float.abs (p -. e))
    done
  done;
  Fmt.pr "max_abs_diff=%g@." !max_diff;
  [%expect {| max_abs_diff=1.19209e-08 |}]

(* Pins one gate-step's raw expression size: with no expression-level sharing
   yet, [tanh_v]'s sign-based predecessor measured 20104 nodes here (see the
   comment on [sigmoid] above) purely from repeated argument embedding --
   almost 30x this. A regression in that direction would make even a single
   layer/direction brush [Kernel.Limits.default]'s 4096 [max_size], and
   Q*R > 1 layers/directions each add their own scan body to the same
   program's total. *)
let%expect_test "one_step's raw expression size stays small without sharing" =
  let dummy_sig = fresh_source "dummy" [| 0. |] (vec_shape ~n:1) in
  let previous_at (_ : Role.Position.t Index.t) = Value.const 0. in
  let layer_input ~step:_ ~col:_ =
    Region_context.load dummy_sig (vec_coord Index.zero)
  in
  let body =
    Expr.Builder.run
      (one_step ~weight_hh:dummy_sig ~weight_ih:dummy_sig
         ~bias_hh:(Some dummy_sig) ~bias_ih:(Some dummy_sig) ~layer_input
         ~step:Index.zero ~lane:Index.zero ~previous_at)
  in
  Fmt.pr "size=%d depth=%d@." (Expr.Fold.size body) (Expr.Fold.depth body);
  [%expect {| size=709 depth=33 |}]
