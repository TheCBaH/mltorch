(* `aten.lstm.input`'s Native form. WORK IN PROGRESS (project step 13 /
   M3+M3b): stacked layers (Q>=1) are supported; bidirectionality (a
   [Layer.reverse]) and batch-first layout are validated-but-rejected with a
   typed diagnostic, not silently mishandled -- see [Shape_error.Lstm]. No
   LSTM support claim is made by this file alone; see
   _ai_/project_todo.md step 13.

   Tensor layout (lstm-plan.md §2's table, time-first only so far):
   input/output on [H=seq, W=batch, C=channel]; h0/c0/h_n/c_n on
   [H=layer*R+direction, W=batch, C=hidden] -- fixed regardless of layout
   (R=1 here, so H=layer). weight_ih/weight_hh pack all 4 gates' rows on
   [N=row, C=col], matching [Linear]'s [Out,1,1,1,1,In] convention;
   bias_ih/bias_hh pack them on [N=row], scalar per row. Layer 0 reads the
   raw input width; every later layer reads the previous layer's [R=1]
   hidden trace, so [I_0=input_size], [I_q=hidden_size] for [q>0]. *)

module Lstm = struct
  (* One direction's parameters: layer 0's forward, or (once supported) any
     layer's reverse. *)
  module Direction = struct
    type t = {
      weight_ih : Tensor_ref.t;
      weight_hh : Tensor_ref.t;
      bias : (Tensor_ref.t * Tensor_ref.t) option;
    }

    let name = "Lstm_direction"

    let jsont : t Jsont.t =
      Jsont.map ~kind:name
        ~dec:(fun json ->
          let ms = Json_util.req_obj json name in
          let get k c = Json_util.req_field ms k c name in
          let bias_ih = Json_util.opt_field ms "bias_ih" Tensor_ref.jsont in
          let bias_hh = Json_util.opt_field ms "bias_hh" Tensor_ref.jsont in
          {
            weight_ih = get "weight_ih" Tensor_ref.jsont;
            weight_hh = get "weight_hh" Tensor_ref.jsont;
            bias =
              (match (bias_ih, bias_hh) with
              | Some bi, Some bh -> Some (bi, bh)
              | _ -> None);
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
               ("weight_ih", ref_ t.weight_ih); ("weight_hh", ref_ t.weight_hh);
             ]
            @ bias_kv))
        Jsont.json

    let operands (t : t) =
      [ t.weight_ih; t.weight_hh ]
      @ match t.bias with None -> [] | Some (bi, bh) -> [ bi; bh ]

    let map_operands f (t : t) =
      {
        weight_ih = f t.weight_ih;
        weight_hh = f t.weight_hh;
        bias = Option.map (fun (bi, bh) -> (f bi, f bh)) t.bias;
      }

    let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
      Fmt.pf fmt "@[<hv 2>{weight_ih=%a;@ weight_hh=%a;@ bias=%a}@]" pp_ref
        t.weight_ih pp_ref t.weight_hh
        (Fmt.option ~none:(Fmt.any "none") (fun fmt (bi, bh) ->
             Fmt.pf fmt "(%a,%a)" pp_ref bi pp_ref bh))
        t.bias
  end

  module Layer = struct
    type t = { forward : Direction.t; reverse : Direction.t option }

    let name = "Lstm_layer"

    let jsont : t Jsont.t =
      Jsont.map ~kind:name
        ~dec:(fun json ->
          let ms = Json_util.req_obj json name in
          {
            forward = Json_util.req_field ms "forward" Direction.jsont name;
            reverse = Json_util.opt_field ms "reverse" Direction.jsont;
          })
        ~enc:(fun t ->
          let reverse_kv =
            match t.reverse with
            | None -> []
            | Some d -> [ ("reverse", Json_util.enc Direction.jsont d) ]
          in
          Json_util.jobj
            (("forward", Json_util.enc Direction.jsont t.forward) :: reverse_kv))
        Jsont.json

    let operands (t : t) =
      Direction.operands t.forward
      @ match t.reverse with None -> [] | Some d -> Direction.operands d

    let map_operands f (t : t) =
      {
        forward = Direction.map_operands f t.forward;
        reverse = Option.map (Direction.map_operands f) t.reverse;
      }

    let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
      Fmt.pf fmt "@[<hv 2>{forward=%a;@ reverse=%a}@]" (Direction.pp pp_ref)
        t.forward
        (Fmt.option ~none:(Fmt.any "none") (Direction.pp pp_ref))
        t.reverse
  end

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
    layers : Layer.t list;
    input : Tensor_ref.t;
    h0 : Tensor_ref.t;
    c0 : Tensor_ref.t;
  }

  let name = "Lstm"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        {
          params = get "params" params_jsont;
          layers = get "layers" (Jsont.list Layer.jsont);
          input = get "input" Tensor_ref.jsont;
          h0 = get "h0" Tensor_ref.jsont;
          c0 = get "c0" Tensor_ref.jsont;
        })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ( "layers",
              Json_util.jarr (List.map (Json_util.enc Layer.jsont) t.layers) );
            ("input", ref_ t.input);
            ("h0", ref_ t.h0);
            ("c0", ref_ t.c0);
          ])
      Jsont.json

  let operands (t : t) =
    [ t.input ] @ List.concat_map Layer.operands t.layers @ [ t.h0; t.c0 ]

  let map_operands f (t : t) =
    {
      t with
      input = f t.input;
      layers = List.map (Layer.map_operands f) t.layers;
      h0 = f t.h0;
      c0 = f t.c0;
    }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>lstm@ input=%a@ layers=%a@ h0=%a@ c0=%a@ params=%a@]"
      pp_ref t.input
      (Fmt.brackets (Fmt.list ~sep:Fmt.comma (Layer.pp pp_ref)))
      t.layers pp_ref t.h0 pp_ref t.c0 pp_params t.params

  let state_shape ~(p : params) ~layers ~batch =
    Vec6.shape ~n:1 ~t:1 ~d:1 ~h:layers ~w:batch ~c:p.hidden_size

  let weight_ih_shape (p : params) ~layer_input_size =
    Vec6.shape ~n:(4 * p.hidden_size) ~t:1 ~d:1 ~h:1 ~w:1 ~c:layer_input_size

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

  (* One direction's resolved operand shapes, the counterpart of
     [Direction.t] with tensor refs replaced by their signatures' shapes
     (what [Region_computation.built] has after resolving operands). *)
  module Direction_shapes = struct
    type t = {
      weight_ih : Vec6.shape;
      weight_hh : Vec6.shape;
      bias : (Vec6.shape * Vec6.shape) option;
    }
  end

  module Layer_shapes = struct
    type t = {
      forward : Direction_shapes.t;
      reverse : Direction_shapes.t option;
    }
  end

  let check_direction ~(p : params) ~layer_input_size (d : Direction_shapes.t) =
    let open Err.Syntax in
    let* () =
      check_operand ~operand:`Lstm_weight_ih
        ~expected:(weight_ih_shape p ~layer_input_size)
        ~actual:d.weight_ih
    in
    let* () =
      check_operand ~operand:`Lstm_weight_hh ~expected:(weight_hh_shape p)
        ~actual:d.weight_hh
    in
    match d.bias with
    | None -> Err.return ()
    | Some (bias_ih, bias_hh) ->
        let expected = bias_shape p in
        let* () = check_operand ~operand:`Lstm_bias ~expected ~actual:bias_ih in
        check_operand ~operand:`Lstm_bias ~expected ~actual:bias_hh

  (* Validates every operand's shape against [params]/[input_shape] and
     returns [(output_shape, h_n_shape, c_n_shape)] -- ordinals 0, 1, 2.
     [R=1] only: a present [Layer_shapes.reverse] is rejected explicitly
     (bidirectionality is unimplemented arithmetic, not an unchecked gap),
     so [h_n]/[c_n] carry one row per layer -- would be [Q*R] once it
     lands. *)
  let output_shape (p : params) ~(input_shape : Vec6.shape)
      ~(layers : Layer_shapes.t list) ~h0_shape ~c0_shape =
    let open Err.Syntax in
    let* () = check_input_layout ~input_shape in
    let batch = Dim.to_int (Vec6.get input_shape Axis.W) in
    let seq = Dim.to_int (Vec6.get input_shape Axis.H) in
    let* () =
      if layers = [] then Err.fail (`Lstm Shape_error.Lstm.Empty_layers)
      else Err.return ()
    in
    let* () =
      Err.List.iter
        (fun (q, (layer : Layer_shapes.t)) ->
          let* () =
            match layer.reverse with
            | None -> Err.return ()
            | Some _ -> Err.fail (`Lstm Shape_error.Lstm.Reverse_unsupported)
          in
          let layer_input_size =
            if q = 0 then p.input_size else p.hidden_size
          in
          check_direction ~p ~layer_input_size layer.forward)
        (List.mapi (fun i l -> (i, l)) layers)
    in
    let num_layers = List.length layers in
    let expected_state = state_shape ~p ~layers:num_layers ~batch in
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

  (* One resolved direction's operands, the [Region_computation]-boundary
     counterpart of [Direction.t] with tensor refs resolved to
     [Tensor_sig.t] (matching [Direction_shapes.t]'s shape-only twin). *)
  module Direction_operands = struct
    type t = {
      weight_ih : Tensor_sig.t;
      weight_hh : Tensor_sig.t;
      bias : (Tensor_sig.t * Tensor_sig.t) option;
    }
  end

  (* Authoritative declarative Region computation: one [width=2*K] scan per
     layer (lanes [0,K) = h, [K,2K) = c -- see the file header), chained so
     each later layer's scan reads the previous layer's completed trace
     (lstm-plan.md §4's dependency between ordinary Region locals and later
     scan bodies), read three different ways for the three output
     ordinals. Every output ordinal gets its OWN independent copy of every
     layer's scan (lstm-plan.md §4's accepted explicit constant-factor
     cost), and [R=1] only: no reverse-direction reads exist yet. *)
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

    (* One completed layer's cached, O(1)-per-call trace reader
       ([Region_program.Builder.scan]'s own [scan_read], forward direction
       only -- [R=1]). *)
    type read_fn =
      row:Expr.Role.Position.t Expr.Index.t ->
      lane:Expr.Role.Position.t Expr.Index.t ->
      Expr.Value.t

    (* Builds layer [q]'s scan, then recurses for the remaining layers with
       this layer's reader available as the next one's [layer_input],
       finally handing every completed layer's reader (in layer order) to
       [finish]. [prev_read] is [None] for layer 0 (read the original
       [input] instead) and [Some] the previous layer's reader otherwise;
       both directions read forward rows since [R=1] here. *)
    let rec build_layers ~scan_limits ~k ~seq ~batch ~input_size ~input ~h0 ~c0
        ~(layers : (int * Direction_operands.t) list) ~prev_read ~acc
        (finish :
          read_fn list ->
          (Region_program.t, Region_program.error) Err.t
          Region_program.Builder.t) :
        (Region_program.t, Region_program.error) Err.t Region_program.Builder.t
        =
      match layers with
      | [] -> finish (List.rev acc)
      | (q, dir) :: rest ->
          let layer_input ~step ~col =
            match prev_read with
            | None ->
                Region_context.load input
                  (Expr.Coord.make ~n:Expr.Index.zero ~t:Expr.Index.zero
                     ~d:Expr.Index.zero ~h:step ~w:batch ~c:col)
            | Some prev_read ->
                let row =
                  Expr.Index.clamp_low
                    (Expr.Index.add
                       (Expr.Index.of_position step)
                       (Expr.Index.const 1))
                in
                prev_read ~row ~lane:col
          in
          let this_layer_input_size = if q = 0 then input_size else k in
          let init ~lane =
            let k_idx = mod_k ~k (Expr.Index.of_position lane) in
            let is_h =
              Expr.Bool.value_lt
                (Expr.Value.value_of_index (Expr.Index.of_position lane))
                (Expr.Value.const (float_of_int k))
            in
            let state_coord =
              Expr.Coord.make ~n:Expr.Index.zero ~t:Expr.Index.zero
                ~d:Expr.Index.zero
                ~h:(Expr.Index.clamp_low (Expr.Index.const q))
                ~w:batch ~c:k_idx
            in
            Expr.Builder.return
              (Expr.Value.select is_h
                 (Region_context.load h0 state_coord)
                 (Region_context.load c0 state_coord))
          in
          let update ~step ~lane ~previous_at =
            one_step ~k ~input_size:this_layer_input_size
              ~weight_hh:dir.Direction_operands.weight_hh
              ~weight_ih:dir.Direction_operands.weight_ih
              ~bias:dir.Direction_operands.bias
              ~layer_input:(fun col -> layer_input ~step ~col)
              ~lane ~previous_at
          in
          Region_program.Builder.scan ~limits:scan_limits ~width:(2 * k)
            ~steps:seq ~init ~update (fun scan_read ->
              build_layers ~scan_limits ~k ~seq ~batch ~input_size ~input ~h0
                ~c0 ~layers:rest ~prev_read:(Some scan_read)
                ~acc:(scan_read :: acc) finish)

    let program ~(limits : Kernel.Limits.t) (p : params) ~output
        ~(layers : Direction_operands.t list) ~(input : Tensor_sig.t) ~h0 ~c0 =
      let open Err.Syntax in
      let k = p.hidden_size in
      let seq = Dim.to_int (Vec6.get input.Tensor_sig.shape Axis.H) in
      let num_layers = List.length layers in
      let scan_limits = Kernel.Limits.scan_limits limits in
      let* partition = Region_context.partition [ Axis.H; Axis.C ] in
      let batch = Expr.Index.output Axis.W in
      let indexed = List.mapi (fun i l -> (i, l)) layers in
      Region_context.program
        (Region_program.Builder.run
           (build_layers ~scan_limits ~k ~seq ~batch ~input_size:p.input_size
              ~input ~h0 ~c0 ~layers:indexed ~prev_read:None ~acc:[]
              (fun reads ->
                let by_layer = Array.of_list reads in
                let output_expr =
                  if output = 0 then
                    let last = by_layer.(num_layers - 1) in
                    let row =
                      Expr.Index.clamp_low
                        (Expr.Index.add
                           (Expr.Index.of_position (Expr.Index.output Axis.H))
                           (Expr.Index.const 1))
                    in
                    last ~row ~lane:(Expr.Index.output Axis.C)
                  else
                    let row = Expr.Index.clamp_low (Expr.Index.const seq) in
                    let lane =
                      if output = 1 then Expr.Index.output Axis.C
                      else
                        Expr.Index.clamp_low
                          (Expr.Index.add (Expr.Index.const k)
                             (Expr.Index.of_position (Expr.Index.output Axis.C)))
                    in
                    let h_val =
                      Expr.Value.value_of_index
                        (Expr.Index.of_position (Expr.Index.output Axis.H))
                    in
                    let rec chain q =
                      let read = by_layer.(q) in
                      if q = num_layers - 1 then read ~row ~lane
                      else
                        Expr.Value.select
                          (Expr.Bool.value_lt h_val
                             (Expr.Value.const (float_of_int (q + 1))))
                          (read ~row ~lane)
                          (chain (q + 1))
                    in
                    chain 0
                in
                Region_program.Builder.finish
                  ~max_size:limits.Kernel.Limits.max_size
                  ~max_depth:limits.Kernel.Limits.max_depth ~partition
                  ~output:output_expr)))
  end
end
