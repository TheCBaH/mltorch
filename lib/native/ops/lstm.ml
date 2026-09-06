(* `aten.lstm.input`'s Native form. WORK IN PROGRESS (project step 13 / M3):
   this landing covers exactly one layer, one direction (forward), time-first
   layout, with an optional bias pair -- the single-layer/direction core the
   plan's own M3/M3b split calls out before stacking, bidirectionality and
   batch-first are added. No LSTM support claim is made by this file alone;
   see _ai_/project_todo.md step 13.

   Tensor layout (lstm-plan.md §2's table, time-first only so far):
   input/output on [H=seq, W=batch, C=channel]; h0/c0/h_n/c_n on
   [H=layer*R+direction (=0 here), W=batch, C=hidden] -- fixed regardless of
   layout. weight_ih/weight_hh pack all 4 gates' rows on [N=row, C=col],
   matching [Linear]'s [Out,1,1,1,1,In] convention; bias_ih/bias_hh pack them
   on [N=row], scalar per row. *)

module Lstm = struct
  type params = { hidden_size : int; input_size : int }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"lstm_params" (fun hidden_size input_size ->
        { hidden_size; input_size })
    |> Jsont.Object.mem "hidden_size" Jsont.int ~enc:(fun p -> p.hidden_size)
    |> Jsont.Object.mem "input_size" Jsont.int ~enc:(fun p -> p.input_size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{hidden_size=%d;@ input_size=%d}@]" p.hidden_size
      p.input_size

  type t = {
    params : params;
    input : Tensor_ref.t;
    weight_ih : Tensor_ref.t;
    weight_hh : Tensor_ref.t;
    bias : (Tensor_ref.t * Tensor_ref.t) option;
    h0 : Tensor_ref.t;
    c0 : Tensor_ref.t;
  }

  let name = "Lstm"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        let bias_ih = Json_util.opt_field ms "bias_ih" Tensor_ref.jsont in
        let bias_hh = Json_util.opt_field ms "bias_hh" Tensor_ref.jsont in
        {
          params = get "params" params_jsont;
          input = get "input" Tensor_ref.jsont;
          weight_ih = get "weight_ih" Tensor_ref.jsont;
          weight_hh = get "weight_hh" Tensor_ref.jsont;
          bias =
            (match (bias_ih, bias_hh) with
            | Some bi, Some bh -> Some (bi, bh)
            | _ -> None);
          h0 = get "h0" Tensor_ref.jsont;
          c0 = get "c0" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        let bias_kv =
          match t.bias with
          | None -> []
          | Some (bi, bh) -> [ ("bias_ih", ref_ bi); ("bias_hh", ref_ bh) ]
        in
        Json_util.jobj
          ([
             ("params", Json_util.enc params_jsont t.params);
             ("input", ref_ t.input);
             ("weight_ih", ref_ t.weight_ih);
             ("weight_hh", ref_ t.weight_hh);
             ("h0", ref_ t.h0);
             ("c0", ref_ t.c0);
           ]
          @ bias_kv))
      Jsont.json

  let operands (t : t) =
    [ t.input; t.weight_ih; t.weight_hh; t.h0; t.c0 ]
    @ match t.bias with None -> [] | Some (bi, bh) -> [ bi; bh ]

  let map_operands f (t : t) =
    {
      t with
      input = f t.input;
      weight_ih = f t.weight_ih;
      weight_hh = f t.weight_hh;
      h0 = f t.h0;
      c0 = f t.c0;
      bias = Option.map (fun (bi, bh) -> (f bi, f bh)) t.bias;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt
      "@[<hv 2>lstm@ input=%a@ weight_ih=%a@ weight_hh=%a@ bias=%a@ h0=%a@ \
       c0=%a@ params=%a@]"
      pp_ref t.input pp_ref t.weight_ih pp_ref t.weight_hh
      (Fmt.option ~none:(Fmt.any "none") (fun fmt (bi, bh) ->
           Fmt.pf fmt "(%a,%a)" pp_ref bi pp_ref bh))
      t.bias pp_ref t.h0 pp_ref t.c0 pp_params t.params

  let state_shape ~(p : params) ~batch =
    Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:p.hidden_size

  let weight_ih_shape (p : params) =
    Vec6.shape ~n:(4 * p.hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:p.input_size

  let weight_hh_shape (p : params) =
    Vec6.shape ~n:(4 * p.hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:p.hidden_size

  let bias_shape (p : params) =
    Vec6.shape ~n:(4 * p.hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:1

  let same_shape a b =
    List.for_all
      (fun axis -> Dim.equal (Vec6.get a axis) (Vec6.get b axis))
      Axis.all

  let check_operand ~operand ~expected ~actual =
    if same_shape expected actual then Err.return ()
    else
      Err.fail
        (`Operand_shape Shape_error.Operand_shape.{ operand; expected; actual })

  (* Time-first only: [N=T=D=1], seq on [H], batch on [W], channel on [C]. *)
  let check_input_layout ~(input_shape : Vec6.shape) =
    let expected =
      Vec6.shape ~n:1 ~t:1 ~d:1
        ~h:(Dim.to_int (Vec6.get input_shape Axis.H))
        ~w:(Dim.to_int (Vec6.get input_shape Axis.W))
        ~c:(Dim.to_int (Vec6.get input_shape Axis.C))
    in
    check_operand ~operand:`Lstm_input ~expected ~actual:input_shape

  (* Validates every operand's shape against [params]/[input_shape] and
     returns [(output_shape, h_n_shape, c_n_shape)] -- ordinals 0, 1, 2. One
     layer, one direction: [h_n]/[c_n] carry a single [H=1] row (would be
     [H=Q*R] once stacking/bidirectionality land). *)
  let output_shape (p : params) ~(input_shape : Vec6.shape) ~weight_ih_shape:wih
      ~weight_hh_shape:whh ~bias_shapes ~h0_shape ~c0_shape =
    let open Err.Syntax in
    let* () = check_input_layout ~input_shape in
    let batch = Dim.to_int (Vec6.get input_shape Axis.W) in
    let seq = Dim.to_int (Vec6.get input_shape Axis.H) in
    let* () =
      check_operand ~operand:`Lstm_weight_ih ~expected:(weight_ih_shape p)
        ~actual:wih
    in
    let* () =
      check_operand ~operand:`Lstm_weight_hh ~expected:(weight_hh_shape p)
        ~actual:whh
    in
    let* () =
      match bias_shapes with
      | None -> Err.return ()
      | Some (bias_ih, bias_hh) ->
          let expected = bias_shape p in
          let* () =
            check_operand ~operand:`Lstm_bias ~expected ~actual:bias_ih
          in
          check_operand ~operand:`Lstm_bias ~expected ~actual:bias_hh
    in
    let expected_state = state_shape ~p ~batch in
    let* () =
      check_operand ~operand:`Lstm_state ~expected:expected_state
        ~actual:h0_shape
    in
    let+ () =
      check_operand ~operand:`Lstm_state ~expected:expected_state
        ~actual:c0_shape
    in
    let out_shape =
      Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:p.hidden_size
    in
    (out_shape, expected_state, expected_state)

  (* Authoritative declarative Region computation: one shared [width=2*K]
     scan (lanes [0,K) = h, [K,2K) = c -- see the file header), read three
     different ways for the three output ordinals. lstm-plan.md §4 accepts
     this as an explicit constant-factor cost ("computing separate scans for
     the three outputs... not a reason to redesign Kernel values in this
     slice"): each output ordinal gets its own independent program, none
     shared with the others yet. *)
  module Computation = struct
    let gate_i = 0
    and gate_f = 1
    and gate_g = 2
    and gate_o = 3

    let unsafe_floor_div_pos x d =
      match Expr.Index.floor_div_pos x d with
      | Ok v -> v
      | Error _ -> assert false

    (* The shared divmod helper (lstm-implementation-plan.md "Stage 2+"):
       [x mod k], encoded from [Add]/[Scale]/[Floor_div_pos] and converted
       back with [Clamp_low] -- sound because [x >= 0] and [k > 0] make the
       remainder provably in [0, k). *)
    let mod_k ~k x =
      let q = unsafe_floor_div_pos x k in
      Expr.Index.clamp_low (Expr.Index.add x (Expr.Index.scale (-k) q))

    let row_index ~k ~gate k_idx =
      Expr.Index.clamp_low
        (Expr.Index.add
           (Expr.Index.const (gate * k))
           (Expr.Index.of_position k_idx))

    let mat_coord ~row ~col =
      Expr.Coord.make ~n:row ~t:Expr.Index.zero ~d:Expr.Index.zero
        ~h:Expr.Index.zero ~w:Expr.Index.zero ~c:col

    let vec_coord row =
      Expr.Coord.make ~n:row ~t:Expr.Index.zero ~d:Expr.Index.zero
        ~h:Expr.Index.zero ~w:Expr.Index.zero ~c:Expr.Index.zero

    (* Sum over [0,extent) of [f j], one Region reduction node -- the AST
       stays the same size whatever [extent] is at runtime. *)
    let sum_over extent f =
      let open Expr.Builder.Syntax in
      Expr.Builder.reduction ~kind:Expr.Reduction.Sum ~lo:Expr.Index.zero
        ~hi:(Expr.Index.const extent) (fun j ->
          let* v = f j in
          Expr.Builder.return v)

    (* One gate's pre-activation: sum_j W_hh[gate,k,j]*prev_h[j]
       + sum_j W_ih[gate,k,j]*layer_input(j) [+ b_hh[gate,k] + b_ih[gate,k]]. *)
    let gate_sum ~k ~input_size ~weight_hh ~weight_ih ~bias ~previous_at
        ~layer_input ~gate k_idx =
      let open Expr.Builder.Syntax in
      let row = row_index ~k ~gate k_idx in
      let* hh =
        sum_over k (fun j ->
            Expr.Builder.return
              (Expr.Value.mul (previous_at j)
                 (Region_context.load weight_hh (mat_coord ~row ~col:j))))
      in
      let* ih =
        sum_over input_size (fun j ->
            Expr.Builder.return
              (Expr.Value.mul (layer_input j)
                 (Region_context.load weight_ih (mat_coord ~row ~col:j))))
      in
      let base = Expr.Value.add hh ih in
      let base =
        match bias with
        | None -> base
        | Some (bias_ih, bias_hh) ->
            Expr.Value.add
              (Expr.Value.add base
                 (Region_context.load bias_hh (vec_coord row)))
              (Region_context.load bias_ih (vec_coord row))
      in
      Expr.Builder.return base

    let neg x = Expr.Value.sub (Expr.Value.const 0.) x

    (* [x] is embedded only once: with no expression-level sharing yet, a
       formula referencing its argument twice or more duplicates that
       argument's whole subtree at every reference -- see [tanh_v]. *)
    let sigmoid x =
      Expr.Value.div (Expr.Value.const 1.)
        (Expr.Value.add (Expr.Value.const 1.) (Expr.Value.exp (neg x)))

    (* [tanh(x) = 2*sigmoid(2x) - 1] -- algebraically the stable form
       lstm-plan.md §5 asks for ([exp(-2x)] either overflows to [+inf] or
       underflows to [0], both of which [1/(1+_)] handles exactly, never the
       [inf/inf] NaN a direct [(e^2x-1)/(e^2x+1)] quotient produces for large
       positive [x]), reached through the identity instead of an explicit
       sign branch so [x] is embedded once, not three times -- avoiding the
       ~30x size compounding a sign/abs-based tanh produced when nested
       (once for the [g] gate, again for [next_c]) and reused at the final
       h/c select (test/native/lstm_region_prototype_test.ml). *)
    let tanh_v x =
      Expr.Value.sub
        (Expr.Value.mul (Expr.Value.const 2.)
           (sigmoid (Expr.Value.mul (Expr.Value.const 2.) x)))
        (Expr.Value.const 1.)

    (* One recurrence step for every lane of the packed [2*K] state, per
       lstm-plan.md §5's ATen gate order i,f,g,o. *)
    let one_step ~k ~input_size ~weight_hh ~weight_ih ~bias ~layer_input ~lane
        ~previous_at =
      let open Expr.Builder.Syntax in
      let k_idx = mod_k ~k (Expr.Index.of_position lane) in
      let* a_i =
        gate_sum ~k ~input_size ~weight_hh ~weight_ih ~bias ~previous_at
          ~layer_input ~gate:gate_i k_idx
      in
      let* a_f =
        gate_sum ~k ~input_size ~weight_hh ~weight_ih ~bias ~previous_at
          ~layer_input ~gate:gate_f k_idx
      in
      let* a_g =
        gate_sum ~k ~input_size ~weight_hh ~weight_ih ~bias ~previous_at
          ~layer_input ~gate:gate_g k_idx
      in
      let* a_o =
        gate_sum ~k ~input_size ~weight_hh ~weight_ih ~bias ~previous_at
          ~layer_input ~gate:gate_o k_idx
      in
      let i = sigmoid a_i
      and f = sigmoid a_f
      and g = tanh_v a_g
      and o = sigmoid a_o in
      let previous_c =
        previous_at
          (Expr.Index.clamp_low
             (Expr.Index.add (Expr.Index.const k)
                (Expr.Index.of_position k_idx)))
      in
      let next_c =
        Expr.Value.add (Expr.Value.mul f previous_c) (Expr.Value.mul i g)
      in
      let next_h = Expr.Value.mul o (tanh_v next_c) in
      let lane_val = Expr.Value.value_of_index (Expr.Index.of_position lane) in
      Expr.Builder.return
        (Expr.Value.select
           (Expr.Bool.value_lt lane_val (Expr.Value.const (float_of_int k)))
           next_h next_c)

    let program ~(limits : Kernel.Limits.t) (p : params) ~output ~weight_ih
        ~weight_hh ~bias ~(input : Tensor_sig.t) ~h0 ~c0 =
      let open Err.Syntax in
      let k = p.hidden_size and input_size = p.input_size in
      let seq = Dim.to_int (Vec6.get input.Tensor_sig.shape Axis.H) in
      let scan_limits = Kernel.Limits.scan_limits limits in
      let* partition = Region_context.partition [ Axis.H; Axis.C ] in
      let batch = Expr.Index.output Axis.W in
      let layer_input ~step ~col =
        Region_context.load input
          (Expr.Coord.make ~n:Expr.Index.zero ~t:Expr.Index.zero
             ~d:Expr.Index.zero ~h:step ~w:batch ~c:col)
      in
      let init ~lane =
        let k_idx = mod_k ~k (Expr.Index.of_position lane) in
        let is_h =
          Expr.Bool.value_lt
            (Expr.Value.value_of_index (Expr.Index.of_position lane))
            (Expr.Value.const (float_of_int k))
        in
        let state_coord =
          Expr.Coord.make ~n:Expr.Index.zero ~t:Expr.Index.zero
            ~d:Expr.Index.zero ~h:Expr.Index.zero ~w:batch ~c:k_idx
        in
        Expr.Builder.return
          (Expr.Value.select is_h
             (Region_context.load h0 state_coord)
             (Region_context.load c0 state_coord))
      in
      let update ~step ~lane ~previous_at =
        one_step ~k ~input_size ~weight_hh ~weight_ih ~bias
          ~layer_input:(fun col -> layer_input ~step ~col)
          ~lane ~previous_at
      in
      let row_final = Expr.Index.clamp_low (Expr.Index.const seq) in
      let lane_output = Expr.Index.output Axis.C in
      Region_context.program
        (Region_program.Builder.run
           (Region_program.Builder.scan ~limits:scan_limits ~width:(2 * k)
              ~steps:seq ~init ~update (fun scan_read ->
                let row, lane =
                  if output = 0 then
                    ( Expr.Index.clamp_low
                        (Expr.Index.add
                           (Expr.Index.of_position (Expr.Index.output Axis.H))
                           (Expr.Index.const 1)),
                      lane_output )
                  else if output = 1 then (row_final, lane_output)
                  else
                    ( row_final,
                      Expr.Index.clamp_low
                        (Expr.Index.add (Expr.Index.const k)
                           (Expr.Index.of_position lane_output)) )
                in
                Region_program.Builder.finish
                  ~max_size:limits.Kernel.Limits.max_size
                  ~max_depth:limits.Kernel.Limits.max_depth ~partition
                  ~output:(scan_read ~row ~lane))))
  end
end
