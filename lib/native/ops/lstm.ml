(* `aten.lstm.input`'s Native form. WORK IN PROGRESS (project step 13 /
   M3+M3b): stacked layers (Q>=1), bidirectionality (R=1 or 2, uniform
   across layers), and both input layouts (batch-first/time-first) are
   supported. No LSTM support claim is made by this file alone -- training,
   packed/unbatched input, projections and dtype/config validation beyond
   basic shape checks remain out of scope (M4's importer boundary); see
   _ai_/project_todo.md step 13.

   Tensor layout (lstm-plan.md §2's table): time-first input/output use
   [H=seq, W=batch, C=channel]; batch-first swaps [H]/[W] ([time_axis]/
   [batch_axis] below). [output]'s C is [R*hidden_size]. h0/c0/h_n/c_n
   always use [H=layer*R+direction, W=batch, C=hidden], regardless of
   layout. weight_ih/weight_hh pack all 4 gates' rows on [N=row, C=col],
   matching [Linear]'s [Out,1,1,1,1,In] convention; bias_ih/bias_hh pack
   them on [N=row], scalar per row. Layer 0 reads the raw input width;
   every later layer reads the previous layer's [R]-wide hidden trace
   (forward then reverse, concatenated in original time order), so
   [I_0=input_size], [I_q=R*hidden_size] for [q>0]. *)

module Lstm = struct
  (* One direction's parameters: any layer's forward or reverse. *)
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

  type params = { hidden_size : int; input_size : int; batch_first : bool }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"lstm_params"
      (fun hidden_size input_size batch_first ->
        { hidden_size; input_size; batch_first })
    |> Jsont.Object.mem "hidden_size" Jsont.int ~enc:(fun p -> p.hidden_size)
    |> Jsont.Object.mem "input_size" Jsont.int ~enc:(fun p -> p.input_size)
    |> Jsont.Object.mem "batch_first" Jsont.bool ~enc:(fun p -> p.batch_first)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{hidden_size=%d;@ input_size=%d;@ batch_first=%b}@]"
      p.hidden_size p.input_size p.batch_first

  (* Time-first: seq on H, batch on W. Batch-first: batch on H, seq on W.
     Channel is C either way (lstm-plan.md §2's table). *)
  let time_axis (p : params) = if p.batch_first then Axis.W else Axis.H
  let batch_axis (p : params) = if p.batch_first then Axis.H else Axis.W

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

  (* [N=T=D=1] regardless of layout -- seq/batch's own H-vs-W placement
     ([time_axis]/[batch_axis]) doesn't matter to this check. *)
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
     [R] (the uniform direction count) is read off the FIRST layer's
     [reverse] presence; every other layer must agree
     ([Nonuniform_direction] otherwise, lstm-plan.md §2's "require a
     uniform direction count... across layers"). [h_n]/[c_n] carry
     [Q*R] rows, laid out layer-then-direction (forward before reverse,
     lstm-plan.md §2). *)
  let output_shape (p : params) ~(input_shape : Vec6.shape)
      ~(layers : Layer_shapes.t list) ~h0_shape ~c0_shape =
    let open Err.Syntax in
    let* () = check_input_layout ~input_shape in
    let batch = Dim.to_int (Vec6.get input_shape (batch_axis p)) in
    let seq = Dim.to_int (Vec6.get input_shape (time_axis p)) in
    let* () =
      if layers = [] then Err.fail (`Lstm Shape_error.Lstm.Empty_layers)
      else Err.return ()
    in
    let bidirectional = (List.hd layers).reverse <> None in
    let directions = if bidirectional then 2 else 1 in
    let* () =
      Err.List.iter
        (fun (q, (layer : Layer_shapes.t)) ->
          let* () =
            if layer.reverse <> None <> bidirectional then
              Err.fail (`Lstm Shape_error.Lstm.Nonuniform_direction)
            else Err.return ()
          in
          let layer_input_size =
            if q = 0 then p.input_size else directions * p.hidden_size
          in
          let* () = check_direction ~p ~layer_input_size layer.forward in
          match layer.reverse with
          | None -> Err.return ()
          | Some d -> check_direction ~p ~layer_input_size d)
        (List.mapi (fun i l -> (i, l)) layers)
    in
    let num_layers = List.length layers in
    let expected_state =
      state_shape ~p ~layers:(num_layers * directions) ~batch
    in
    let* () =
      check_operand ~operand:`Lstm_state ~expected:expected_state
        ~actual:h0_shape
    in
    let+ () =
      check_operand ~operand:`Lstm_state ~expected:expected_state
        ~actual:c0_shape
    in
    let out_shape =
      let base =
        Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:(directions * p.hidden_size)
      in
      Vec6.set
        (Vec6.set base (time_axis p) (Dim.extent seq))
        (batch_axis p) (Dim.extent batch)
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

  module Layer_operands = struct
    type t = {
      forward : Direction_operands.t;
      reverse : Direction_operands.t option;
    }
  end

  (* Authoritative declarative Region computation: one [width=2*K] scan per
     layer/direction (lanes [0,K) = h, [K,2K) = c -- see the file header),
     chained so each later layer's scan reads the previous layer's
     completed trace(s) (lstm-plan.md §4's dependency between ordinary
     Region locals and later scan bodies), read three different ways for
     the three output ordinals. Every output ordinal gets its OWN
     independent copy of every layer/direction's scan (lstm-plan.md §4's
     accepted explicit constant-factor cost). *)
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

    (* One completed layer/direction's cached, O(1)-per-call trace reader
       ([Region_program.Builder.scan]'s own [scan_read]). *)
    type read_fn =
      row:Expr.Role.Position.t Expr.Index.t ->
      lane:Expr.Role.Position.t Expr.Index.t ->
      Expr.Value.t

    (* [t]: this scan's own original-time index at recurrence step [step]
       (lstm-plan.md §5) -- [step] itself for forward, [(seq-1)-step] for
       reverse, converted back to a position (sound: both bounds keep it in
       [0,seq)). *)
    let original_time ~seq ~is_reverse step =
      if not is_reverse then step
      else
        Expr.Index.clamp_low
          (Expr.Index.add
             (Expr.Index.const (seq - 1))
             (Expr.Index.scale (-1) (Expr.Index.of_position step)))

    (* Reads the previous layer's completed trace(s) at column [col] of a
       [q>0] layer's [R*K]-wide input, at this scan's own original time [t].
       [R=1] (no [reverse_read]): forward row [t+1], no direction demux
       needed at all. [R=2]: [col<K] selects the previous layer's forward
       trace (row [t+1]); [col>=K] selects its reverse trace (row [seq-t],
       lstm-plan.md §5) -- both at lane [col mod K]. *)
    let read_prev_layer ~k ~seq ~t ~col
        ((forward_read, reverse_read) : read_fn * read_fn option) =
      match reverse_read with
      | None ->
          let row =
            Expr.Index.clamp_low
              (Expr.Index.add (Expr.Index.of_position t) (Expr.Index.const 1))
          in
          forward_read ~row ~lane:col
      | Some reverse_read ->
          let k_idx = mod_k ~k (Expr.Index.of_position col) in
          let is_fwd =
            Expr.Bool.value_lt
              (Expr.Value.value_of_index (Expr.Index.of_position col))
              (Expr.Value.const (float_of_int k))
          in
          let row_fwd =
            Expr.Index.clamp_low
              (Expr.Index.add (Expr.Index.of_position t) (Expr.Index.const 1))
          in
          let row_rev =
            Expr.Index.clamp_low
              (Expr.Index.add (Expr.Index.const seq)
                 (Expr.Index.scale (-1) (Expr.Index.of_position t)))
          in
          Expr.Value.select is_fwd
            (forward_read ~row:row_fwd ~lane:k_idx)
            (reverse_read ~row:row_rev ~lane:k_idx)

    (* Builds one (layer, direction)'s scan. [prev] is [None] for layer 0
       (read the original [input] instead) and [Some] the previous layer's
       (forward_read, reverse_read option) otherwise. [state_row] (static:
       [q*directions+d]) selects this direction's own [h0]/[c0] slice. *)
    let build_one ~scan_limits ~k ~seq ~batch ~time_axis ~batch_axis ~input_size
        ~input ~h0 ~c0 ~is_reverse ~state_row ~prev (dir : Direction_operands.t)
        (continue : read_fn -> 'a Region_program.Builder.t) :
        'a Region_program.Builder.t =
      let layer_input ~step ~col =
        let t = original_time ~seq ~is_reverse step in
        match prev with
        | None ->
            let base =
              Expr.Coord.make ~n:Expr.Index.zero ~t:Expr.Index.zero
                ~d:Expr.Index.zero ~h:Expr.Index.zero ~w:Expr.Index.zero ~c:col
            in
            Region_context.load input
              (Expr.Coord.set
                 (Expr.Coord.set base time_axis t)
                 batch_axis batch)
        | Some prev -> read_prev_layer ~k ~seq ~t ~col prev
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
            ~d:Expr.Index.zero
            ~h:(Expr.Index.clamp_low (Expr.Index.const state_row))
            ~w:batch ~c:k_idx
        in
        Expr.Builder.return
          (Expr.Value.select is_h
             (Region_context.load h0 state_coord)
             (Region_context.load c0 state_coord))
      in
      let update ~step ~lane ~previous_at =
        one_step ~k ~input_size ~weight_hh:dir.Direction_operands.weight_hh
          ~weight_ih:dir.Direction_operands.weight_ih
          ~bias:dir.Direction_operands.bias
          ~layer_input:(fun col -> layer_input ~step ~col)
          ~lane ~previous_at
      in
      Region_program.Builder.scan ~limits:scan_limits ~width:(2 * k) ~steps:seq
        ~init ~update continue

    (* Builds every layer's scan(s), forward before reverse per layer
       (lstm-plan.md §2), accumulating a FLAT [(a, read_fn)] list in [a]
       order ([a = layer*directions+direction]) -- what the [h_n]/[c_n]
       select chain below reads. *)
    let rec build_layers ~scan_limits ~k ~seq ~batch ~time_axis ~batch_axis
        ~directions ~input_size ~input ~h0 ~c0
        ~(layers : (int * Layer_operands.t) list) ~prev ~acc
        (finish :
          (int * read_fn) list ->
          (Region_program.t, Region_program.error) Err.t
          Region_program.Builder.t) :
        (Region_program.t, Region_program.error) Err.t Region_program.Builder.t
        =
      match layers with
      | [] -> finish (List.rev acc)
      | (q, (layer : Layer_operands.t)) :: rest ->
          let this_layer_input_size =
            if q = 0 then input_size else k * directions
          in
          build_one ~scan_limits ~k ~seq ~batch ~time_axis ~batch_axis
            ~input_size:this_layer_input_size ~input ~h0 ~c0 ~is_reverse:false
            ~state_row:(q * directions) ~prev layer.forward (fun forward_read ->
              match layer.reverse with
              | None ->
                  build_layers ~scan_limits ~k ~seq ~batch ~time_axis
                    ~batch_axis ~directions ~input_size ~input ~h0 ~c0
                    ~layers:rest
                    ~prev:(Some (forward_read, None))
                    ~acc:((q * directions, forward_read) :: acc)
                    finish
              | Some reverse ->
                  build_one ~scan_limits ~k ~seq ~batch ~time_axis ~batch_axis
                    ~input_size:this_layer_input_size ~input ~h0 ~c0
                    ~is_reverse:true
                    ~state_row:((q * directions) + 1)
                    ~prev reverse
                    (fun reverse_read ->
                      build_layers ~scan_limits ~k ~seq ~batch ~time_axis
                        ~batch_axis ~directions ~input_size ~input ~h0 ~c0
                        ~layers:rest
                        ~prev:(Some (forward_read, Some reverse_read))
                        ~acc:
                          (((q * directions) + 1, reverse_read)
                          :: (q * directions, forward_read)
                          :: acc)
                        finish))

    let program ~(limits : Kernel.Limits.t) (p : params) ~output
        ~(layers : Layer_operands.t list) ~(input : Tensor_sig.t) ~h0 ~c0 =
      let open Err.Syntax in
      let k = p.hidden_size in
      let time_axis = time_axis p and batch_axis = batch_axis p in
      let seq = Dim.to_int (Vec6.get input.Tensor_sig.shape time_axis) in
      let num_layers = List.length layers in
      let directions =
        if (List.hd layers).Layer_operands.reverse <> None then 2 else 1
      in
      let scan_limits = Kernel.Limits.scan_limits limits in
      (* [output]'s output/h_n/c_n each have their OWN axis meanings
         (lstm-plan.md §4): [output]'s time axis is layout-dependent and
         its batch axis is whichever of H/W is left; [h_n]/[c_n] are
         layout-independent (always [H=layer*R+direction, W=batch]). Each
         ordinal's partition/batch-key axis follows its own layout. *)
      let* partition =
        Region_context.partition
          (if output = 0 then [ time_axis; Axis.C ] else [ Axis.H; Axis.C ])
      in
      let batch =
        Expr.Index.output (if output = 0 then batch_axis else Axis.W)
      in
      let indexed = List.mapi (fun i l -> (i, l)) layers in
      Region_context.program
        (Region_program.Builder.run
           (build_layers ~scan_limits ~k ~seq ~batch ~time_axis ~batch_axis
              ~directions ~input_size:p.input_size ~input ~h0 ~c0
              ~layers:indexed ~prev:None ~acc:[] (fun reads ->
                let by_a =
                  Array.make (num_layers * directions) (fun ~row:_ ~lane:_ ->
                      assert false)
                in
                List.iter (fun (a, read) -> by_a.(a) <- read) reads;
                let output_expr =
                  if output = 0 then
                    let last_layer_first_dir = (num_layers - 1) * directions in
                    let last =
                      ( by_a.(last_layer_first_dir),
                        if directions = 2 then
                          Some by_a.(last_layer_first_dir + 1)
                        else None )
                    in
                    let c_val =
                      Expr.Value.value_of_index
                        (Expr.Index.of_position (Expr.Index.output Axis.C))
                    in
                    let k_idx =
                      mod_k ~k
                        (Expr.Index.of_position (Expr.Index.output Axis.C))
                    in
                    let t_out = Expr.Index.output time_axis in
                    match snd last with
                    | None ->
                        let row =
                          Expr.Index.clamp_low
                            (Expr.Index.add
                               (Expr.Index.of_position t_out)
                               (Expr.Index.const 1))
                        in
                        (fst last) ~row ~lane:k_idx
                    | Some reverse_read ->
                        let is_fwd =
                          Expr.Bool.value_lt c_val
                            (Expr.Value.const (float_of_int k))
                        in
                        let row_fwd =
                          Expr.Index.clamp_low
                            (Expr.Index.add
                               (Expr.Index.of_position t_out)
                               (Expr.Index.const 1))
                        in
                        let row_rev =
                          Expr.Index.clamp_low
                            (Expr.Index.add (Expr.Index.const seq)
                               (Expr.Index.scale (-1)
                                  (Expr.Index.of_position t_out)))
                        in
                        Expr.Value.select is_fwd
                          ((fst last) ~row:row_fwd ~lane:k_idx)
                          (reverse_read ~row:row_rev ~lane:k_idx)
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
                    let total = num_layers * directions in
                    let rec chain a =
                      let read = by_a.(a) in
                      if a = total - 1 then read ~row ~lane
                      else
                        Expr.Value.select
                          (Expr.Bool.value_lt h_val
                             (Expr.Value.const (float_of_int (a + 1))))
                          (read ~row ~lane)
                          (chain (a + 1))
                    in
                    chain 0
                in
                Region_program.Builder.finish
                  ~max_size:limits.Kernel.Limits.max_size
                  ~max_depth:limits.Kernel.Limits.max_depth ~partition
                  ~output:output_expr)))
  end
end
