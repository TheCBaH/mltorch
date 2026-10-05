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

(* A pack of fresh definitions with the types of the template's. *)
let rec fresh_like : type s. t -> s pack -> s pack =
 fun b -> function
  | Nil -> Nil
  | Cons (v, rest) ->
      let w = mint b v.Ssa_value.ty in
      Cons (w, fresh_like b rest)

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
let index_add b x y = one b (Ssa_op.Index_add (x, y))

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
  | Ssa_op.Decode.Bool_to_f64 | Ssa_op.Decode.F32_to_f64 -> ()
  | Ssa_op.Decode.I64 -> invalid_arg "Ssa_builder.load_f64: an i64 decode");
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
let child b ~live = { shared = b.shared; rev = []; live }

let new_region b ~params ~body ~yields =
  let id, next = Ssa_id.Region.Next.alloc b.shared.next_region in
  b.shared.next_region <- next;
  { Ssa_region.id; params; body; yields }

let finish_region b inner ~params ~yields =
  new_region b ~params ~body:(List.rev inner.rev) ~yields

let for_ b ~lo ~hi ~init body =
  let iv = mint b (scalar Ssa_type.Index) in
  let carried = fresh_like b init in
  let eff = mint b Ssa_type.Effect in
  let inner = child b ~live:eff in
  let next = body inner iv carried in
  let region =
    finish_region b inner
      ~params:((iv :: raw_list carried) @ [ eff ])
      ~yields:(raw_list next @ [ inner.live ])
  in
  let results = fresh_like b init in
  let eff_out = mint b Ssa_type.Effect in
  emit b
    (Ssa_stmt.For
       {
         lo;
         hi;
         step = 1L;
         inits = raw_list init @ [ b.live ];
         results = raw_list results @ [ eff_out ];
         body = region;
       });
  b.live <- eff_out;
  results

let if_ b cond ~then_ ~else_ =
  let branch f =
    let inner = child b ~live:b.live in
    let out = f inner in
    ( out,
      finish_region b inner ~params:[] ~yields:(raw_list out @ [ inner.live ])
    )
  in
  let yes, then_region = branch then_ in
  let _, else_region = branch else_ in
  let results = fresh_like b yes in
  let eff_out = mint b Ssa_type.Effect in
  emit b
    (Ssa_stmt.If
       {
         cond;
         results = raw_list results @ [ eff_out ];
         then_ = then_region;
         else_ = else_region;
       });
  b.live <- eff_out;
  results

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

let set_origin b o = b.shared.origin <- o

let program ~buffers f =
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
      next_value = shared.next_value;
      next_region = shared.next_region;
    }
  in
  Err.map (fun () -> p) (Ssa_verify.check p)
