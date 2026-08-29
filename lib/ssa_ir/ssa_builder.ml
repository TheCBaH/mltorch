type 'a value = Ssa_value.t

type _ pack =
  | Nil : unit pack
  | Cons : 'a value * 'rest pack -> ('a * 'rest) pack

type access =
  | Coord of Ssa_type.index value Expr.Coord.t
  | Flat of Ssa_type.index value

type shared = {
  buffers : Ssa_buffer.t list;
  mutable next_value : Ssa_id.Value.Next.t;
  mutable next_region : Ssa_id.Region.Next.t;
  mutable origin : Ssa_origin.t;
}

type t = {
  shared : shared;
  mutable rev : Ssa_region.t Ssa_stmt.t list;
  mutable live : Ssa_value.t;
}

let mint b ty : 'a value =
  let id, next = Ssa_id.Value.Next.alloc b.shared.next_value in
  b.shared.next_value <- next;
  { Ssa_value.id; ty }

let scalar s = Ssa_type.Scalar s
let emit b s = b.rev <- s :: b.rev

let rec raw_list : type s. s pack -> Ssa_value.t list = function
  | Nil -> []
  | Cons (v, rest) -> v :: raw_list rest

let last vs = List.nth vs (List.length vs - 1)

let instr b op =
  let values =
    match Ssa_typing.result_types op with
    | Ok ts -> ts
    | Error e -> invalid_arg (Fmt.str "Ssa_builder: %a" Ssa_typing.pp_error e)
  in
  let effectful = Ssa_op.effectful op in
  let types = if effectful then values @ [ Ssa_type.Effect ] else values in
  let results = List.map (mint b) types in
  let token = if effectful then Some b.live else None in
  emit b
    (Ssa_stmt.Instr { Ssa_instr.results; op; token; origin = b.shared.origin });
  if effectful then b.live <- last results;
  results

let one b op =
  match instr b op with
  | v :: _ -> v
  | [] -> invalid_arg "Ssa_builder: operation has no value result"

let const b c : 'a value = one b (Ssa_op.Const c)

let f32 b x =
  if not (Core.Float_bits.equal_portable x (Ssa_const.round_f32 x)) then
    invalid_arg "Ssa_builder.f32: not representable in binary32";
  const b (Ssa_const.F32 x)

let f64 b x = const b (Ssa_const.F64 x)
let i64 b x = const b (Ssa_const.I64 x)

let index b x =
  if not (Ssa_const.in_index_domain x) then
    invalid_arg "Ssa_builder.index: outside the index domain";
  const b (Ssa_const.Index x)

let pred b x = const b (Ssa_const.Pred x)
let convert b c (a : 'a value) : 'b value = one b (Ssa_op.Convert (c, a))
let f32_to_f64 b a = convert b Ssa_op.Convert.F32_to_f64 a
let f64_to_f32 b a = convert b Ssa_op.Convert.F64_to_f32 a
let index_to_f64 b a = convert b Ssa_op.Convert.Index_to_f64 a
let index_to_i64 b a = convert b Ssa_op.Convert.Index_to_i64 a
let f64_binary b op x y = one b (Ssa_op.Float_binary (op, x, y))
let f64_max b x y = one b (Ssa_op.Float_max (x, y))
let f64_unary b op x = one b (Ssa_op.Float_unary (op, x))
let float_compare b c x y = one b (Ssa_op.Float_compare (c, x, y))
let float_to_i64 b x = one b (Ssa_op.Float_to_i64 x)
let i64_arith b op x y = one b (Ssa_op.I64_arith (op, x, y))
let i64_compare b c x y = one b (Ssa_op.I64_compare (c, x, y))
let i64_div b x y = one b (Ssa_op.I64_div (x, y))
let i64_to_f32 b a = convert b Ssa_op.Convert.I64_to_f32 a
let i64_to_f64 b a = convert b Ssa_op.Convert.I64_to_f64 a
let index_of_i64 b x = one b (Ssa_op.Index_of_i64 x)
let index_add b x y = one b (Ssa_op.Index_add (x, y))
let index_ceil_div b k x = one b (Ssa_op.Index_ceil_div (k, x))
let index_clamp_low b x = one b (Ssa_op.Index_clamp_low x)
let index_compare b c x y = one b (Ssa_op.Index_compare (c, x, y))
let index_floor_div b k x = one b (Ssa_op.Index_floor_div (k, x))
let index_max b x y = one b (Ssa_op.Index_max (x, y))
let index_min b x y = one b (Ssa_op.Index_min (x, y))
let pool_better b best value = one b (Ssa_op.Pool_better (best, value))
let pred_not b x = one b (Ssa_op.Pred_not x)
let pred_or b x y = one b (Ssa_op.Pred_or (x, y))
let select b p x y = one b (Ssa_op.Select (p, x, y))

let index_scale b k x =
  if not (Ssa_const.in_index_domain k) then
    invalid_arg "Ssa_builder.index_scale: literal outside the index domain";
  one b (Ssa_op.Index_scale (k, x))

let access : access -> Ssa_access.t = function
  | Coord c -> Ssa_access.Coord c
  | Flat v -> Ssa_access.Flat v

let declared b id =
  if
    not
      (List.exists
         (fun (x : Ssa_buffer.t) -> Ssa_id.Buffer.equal x.Ssa_buffer.id id)
         b.shared.buffers)
  then
    invalid_arg (Fmt.str "Ssa_builder: %a is not declared" Ssa_id.Buffer.pp id)

let load_f64 b buffer ~decode at =
  declared b buffer;
  (match decode with
  | Ssa_op.Decode.I64 -> invalid_arg "Ssa_builder.load_f64: an i64 decode"
  | Ssa_op.Decode.Bf16_to_f64 | Ssa_op.Decode.Bool_to_f64
  | Ssa_op.Decode.F16_to_f64 | Ssa_op.Decode.F32_to_f64
  | Ssa_op.Decode.F64_to_f64 | Ssa_op.Decode.I16_dequant
  | Ssa_op.Decode.I32_to_f64 | Ssa_op.Decode.I64_to_f64
  | Ssa_op.Decode.I8_dequant ->
      ());
  one b (Ssa_op.Load { buffer; at = access at; decode })

let load_i64 b buffer at =
  declared b buffer;
  one b (Ssa_op.Load { buffer; at = access at; decode = Ssa_op.Decode.I64 })

let store b buffer ~encode at value =
  declared b buffer;
  ignore (instr b (Ssa_op.Store { buffer; at = access at; encode; value }))

let store_f64 b buffer ~encode at value =
  (match encode with
  | Ssa_op.Encode.Bool_nonzero | Ssa_op.Encode.F32_round -> ()
  | Ssa_op.Encode.I64 -> invalid_arg "Ssa_builder.store_f64: an i64 encode");
  store b buffer ~encode at value

let store_i64 b buffer at value =
  store b buffer ~encode:Ssa_op.Encode.I64 at value

let mark b m = ignore (instr b (Ssa_op.Mark m))
let local_alloc ?var b ~slots = one b (Ssa_op.Local_alloc { slots; var })
let local_read b local at = one b (Ssa_op.Local_read { local; at })

let local_write b local at value =
  ignore (instr b (Ssa_op.Local_write { local; at; value }))

let check_local b ~var ~extent at =
  ignore (instr b (Ssa_op.Check_local { var; at; extent }))

let check_scan b ~var ~row ~lane ~row_extent ~lane_extent =
  ignore
    (instr b (Ssa_op.Check_scan { var; row; lane; row_extent; lane_extent }))

let meter_charge b = ignore (instr b Ssa_op.Meter_charge)
let meter_release b ~width = ignore (instr b (Ssa_op.Meter_release width))
let meter_reserve b ~width = ignore (instr b (Ssa_op.Meter_reserve width))
let meter_reset b = ignore (instr b Ssa_op.Meter_reset)

let check_gather b raw ~extent =
  ignore (instr b (Ssa_op.Check_gather { raw; extent }))

let check_access b buffer at =
  declared b buffer;
  ignore (instr b (Ssa_op.Check_access { buffer; at = access at }))

let child b ~live = { shared = b.shared; rev = []; live }

let new_region b ~params ~body ~yields =
  let id, next = Ssa_id.Region.Next.alloc b.shared.next_region in
  b.shared.next_region <- next;
  { Ssa_region.id; params; body; yields }

let finish_region b inner ~params ~yields =
  new_region b ~params ~body:(List.rev inner.rev) ~yields

let signature_error what =
  invalid_arg ("Ssa_builder: " ^ what ^ " do not match the carried signature")

let same_types a b =
  List.length a = List.length b
  && List.for_all2
       (fun (x : Ssa_value.t) (y : Ssa_value.t) ->
         Ssa_type.equal x.Ssa_value.ty y.Ssa_value.ty)
       a b

let fresh_list b vs =
  List.map (fun (v : Ssa_value.t) -> mint b v.Ssa_value.ty) vs

let for_dyn b ~lo ~hi ~init body =
  let iv = mint b (scalar Ssa_type.Index) in
  let carried = fresh_list b init in
  let eff = mint b Ssa_type.Effect in
  let inner = child b ~live:eff in
  let next = body inner iv carried in
  if not (same_types init next) then signature_error "yielded values";
  let region =
    finish_region b inner
      ~params:((iv :: carried) @ [ eff ])
      ~yields:(next @ [ inner.live ])
  in
  let results = fresh_list b init in
  let eff_out = mint b Ssa_type.Effect in
  emit b
    (Ssa_stmt.For
       {
         lo;
         hi;
         step = 1L;
         inits = init @ [ b.live ];
         results = results @ [ eff_out ];
         body = region;
       });
  b.live <- eff_out;
  results

let if_dyn b cond ~then_ ~else_ =
  let branch f =
    let inner = child b ~live:b.live in
    let out = f inner in
    (out, finish_region b inner ~params:[] ~yields:(out @ [ inner.live ]))
  in
  let yes, then_region = branch then_ in
  let no, else_region = branch else_ in
  if not (same_types yes no) then signature_error "branch results";
  let results = fresh_list b yes in
  let eff_out = mint b Ssa_type.Effect in
  emit b
    (Ssa_stmt.If
       {
         cond;
         results = results @ [ eff_out ];
         then_ = then_region;
         else_ = else_region;
       });
  b.live <- eff_out;
  results

(* The pack of a flat list of values, in the shape and types of a template. *)
let rec rebuild : type s. s pack -> Ssa_value.t list -> s pack =
 fun template vs ->
  match (template, vs) with
  | Nil, [] -> Nil
  | Cons (w, rest), v :: vs ->
      if not (Ssa_type.equal w.Ssa_value.ty v.Ssa_value.ty) then
        signature_error "values";
      Cons (v, rebuild rest vs)
  | Nil, _ :: _ | Cons _, [] -> signature_error "values"

let for_ b ~lo ~hi ~init body =
  let results =
    for_dyn b ~lo ~hi ~init:(raw_list init) (fun b iv carried ->
        raw_list (body b iv (rebuild init carried)))
  in
  rebuild init results

let if_ b cond ~then_ ~else_ =
  let template = ref None in
  let results =
    if_dyn b cond
      ~then_:(fun b ->
        let out = then_ b in
        template := Some out;
        raw_list out)
      ~else_:(fun b -> raw_list (else_ b))
  in
  match !template with
  | Some t -> rebuild t results
  | None -> invalid_arg "Ssa_builder.if_: no then-branch"

let as_type ty (v : Ssa_value.t) : 'a value =
  if Ssa_type.equal v.Ssa_value.ty ty then v
  else
    invalid_arg
      (Fmt.str "Ssa_builder: %a is not %a" Ssa_type.pp v.Ssa_value.ty
         Ssa_type.pp ty)

let as_f64 v = as_type (scalar Ssa_type.F64) v
let as_i64 v = as_type (scalar Ssa_type.I64) v
let as_index v = as_type (scalar Ssa_type.Index) v
let as_pred v = as_type (scalar Ssa_type.Pred) v
let as_local v = as_type Ssa_type.Local v

let ordered_sum b ~lo ~hi ~seed body =
  let iv = mint b (scalar Ssa_type.Index) in
  let eff = mint b Ssa_type.Effect in
  let inner = child b ~live:eff in
  let term = body inner iv in
  let region =
    finish_region b inner ~params:[ iv; eff ] ~yields:[ term; inner.live ]
  in
  let sum = mint b seed.Ssa_value.ty in
  let eff_out = mint b Ssa_type.Effect in
  emit b
    (Ssa_stmt.Ordered_sum
       {
         lo;
         hi;
         seed;
         token = b.live;
         results = [ sum; eff_out ];
         body = region;
       });
  b.live <- eff_out;
  sum

(* Runs [f] against a block that is thrown away. It spends value and region ids,
   which only leaves gaps: an id names a definition, it is not an ordinal. *)
let probe b f =
  let inner = child b ~live:(mint b Ssa_type.Effect) in
  ignore (f inner)

let set_origin b o = b.shared.origin <- o

let program ?(scan_limits = Expr.Scan_limits.default) ~buffers f =
  let shared =
    {
      buffers;
      next_value = Ssa_id.Value.Next.first;
      next_region = Ssa_id.Region.Next.first;
      origin = Ssa_origin.Unknown;
    }
  in
  let root =
    {
      shared;
      rev = [];
      live = { Ssa_value.id = Ssa_id.Value.of_int 0; ty = Ssa_type.Effect };
    }
  in
  let entry_effect = mint root Ssa_type.Effect in
  root.live <- entry_effect;
  let b = root in
  f b;
  let entry = finish_region b b ~params:[ entry_effect ] ~yields:[ b.live ] in
  let p =
    {
      Ssa_program.revision = Ssa_id.Revision.of_int 0;
      buffers;
      entry;
      scan_limits;
      next_value = shared.next_value;
      next_region = shared.next_region;
    }
  in
  Err.map (fun () -> p) (Ssa_verify.check p)
