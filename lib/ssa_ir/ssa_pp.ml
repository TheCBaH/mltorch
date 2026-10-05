(* Values are named in the order they are defined while printing. A use that no
   definition has reached prints as [%?] and its id, which is how an ill-formed
   program shows its defect instead of hiding it behind a fresh name. *)

type state = {
  out : Buffer.t;
  names : (int, int) Hashtbl.t;
  mutable next : int;
}

let line st indent fmt =
  Fmt.kstr
    (fun s ->
      Buffer.add_string st.out (String.make (2 * indent) ' ');
      Buffer.add_string st.out s;
      Buffer.add_char st.out '\n')
    fmt

let name st (v : Ssa_value.t) =
  match Hashtbl.find_opt st.names (v.Ssa_value.id :> int) with
  | Some n -> Fmt.str "%%%d" n
  | None -> Fmt.str "%%?%d" (v.Ssa_value.id :> int)

let define st (v : Ssa_value.t) =
  let n = st.next in
  st.next <- n + 1;
  Hashtbl.replace st.names (v.Ssa_value.id :> int) n;
  Fmt.str "%%%d:%a" n Ssa_type.pp v.Ssa_value.ty

let join = String.concat ", "
let uses st vs = join (List.map (name st) vs)
let defs st vs = join (List.map (define st) vs)

let access st = function
  | Ssa_access.Coord c ->
      "[" ^ join (List.map (name st) (Expr.Coord.to_list c)) ^ "]"
  | Ssa_access.Flat v -> "@" ^ name st v

let op st (o : Ssa_op.t) =
  match o with
  | Ssa_op.Check_access { buffer; at } ->
      Fmt.str "check_access %a%s" Ssa_id.Buffer.pp buffer (access st at)
  | Ssa_op.Check_gather { raw; extent } ->
      Fmt.str "check_gather %s, %Ld" (name st raw) extent
  | Ssa_op.Const c -> Fmt.str "const %a" Ssa_const.pp c
  | Ssa_op.Convert (c, a) ->
      Fmt.str "convert.%s %s" (Ssa_op.Convert.name c) (name st a)
  | Ssa_op.Float_binary (b, x, y) ->
      Fmt.str "float.%s %s, %s" (Ssa_op.binary_name b) (name st x) (name st y)
  | Ssa_op.Float_compare (c, x, y) ->
      Fmt.str "float.compare.%s %s, %s" (Ssa_op.Compare.name c) (name st x)
        (name st y)
  | Ssa_op.Float_max (x, y) ->
      Fmt.str "float.max %s, %s" (name st x) (name st y)
  | Ssa_op.Float_to_i64 x -> Fmt.str "float.to_i64 %s" (name st x)
  | Ssa_op.I64_arith (o, x, y) ->
      Fmt.str "i64.%s %s, %s" (Ssa_op.I64_op.name o) (name st x) (name st y)
  | Ssa_op.I64_compare (c, x, y) ->
      Fmt.str "i64.compare.%s %s, %s" (Ssa_op.Compare.name c) (name st x)
        (name st y)
  | Ssa_op.I64_div (x, y) -> Fmt.str "i64.div %s, %s" (name st x) (name st y)
  | Ssa_op.Float_unary (u, x) ->
      Fmt.str "float.%s %s" (Ssa_op.unary_name u) (name st x)
  | Ssa_op.Index_add (x, y) ->
      Fmt.str "index.add %s, %s" (name st x) (name st y)
  | Ssa_op.Index_ceil_div (k, x) ->
      Fmt.str "index.ceil_div %Ld, %s" k (name st x)
  | Ssa_op.Index_clamp_low x -> Fmt.str "index.clamp_low %s" (name st x)
  | Ssa_op.Index_compare (c, x, y) ->
      Fmt.str "index.compare.%s %s, %s" (Ssa_op.Compare.name c) (name st x)
        (name st y)
  | Ssa_op.Index_floor_div (k, x) ->
      Fmt.str "index.floor_div %Ld, %s" k (name st x)
  | Ssa_op.Index_max (x, y) ->
      Fmt.str "index.max %s, %s" (name st x) (name st y)
  | Ssa_op.Index_min (x, y) ->
      Fmt.str "index.min %s, %s" (name st x) (name st y)
  | Ssa_op.Index_of_i64 x -> Fmt.str "index.of_i64 %s" (name st x)
  | Ssa_op.Index_scale (k, x) -> Fmt.str "index.scale %Ld, %s" k (name st x)
  | Ssa_op.Load { buffer; at; decode } ->
      Fmt.str "load.%s %a%s"
        (Ssa_op.Decode.name decode)
        Ssa_id.Buffer.pp buffer (access st at)
  | Ssa_op.Mark m -> Fmt.str "mark %s" (Ssa_mark.name m)
  | Ssa_op.Pool_better (x, y) ->
      Fmt.str "pool_better %s, %s" (name st x) (name st y)
  | Ssa_op.Pred_not x -> Fmt.str "pred.not %s" (name st x)
  | Ssa_op.Pred_or (x, y) -> Fmt.str "pred.or %s, %s" (name st x) (name st y)
  | Ssa_op.Select (p, x, y) ->
      Fmt.str "select %s, %s, %s" (name st p) (name st x) (name st y)
  | Ssa_op.Store { buffer; at; encode; value } ->
      Fmt.str "store.%s %a%s, %s"
        (Ssa_op.Encode.name encode)
        Ssa_id.Buffer.pp buffer (access st at) (name st value)

let results st rs = match rs with [] -> "" | _ -> "(" ^ defs st rs ^ ") = "

let rec region st indent (r : Ssa_region.t) =
  List.iter (stmt st indent) r.Ssa_region.body;
  line st indent "yield %s" (uses st r.Ssa_region.yields)

and params st (r : Ssa_region.t) = defs st r.Ssa_region.params

and stmt st indent : Ssa_region.t Ssa_stmt.t -> unit = function
  | Ssa_stmt.For { lo; hi; step; inits; results = rs; body } ->
      let header_uses =
        Fmt.str "%s to %s step %Ld" (name st lo) (name st hi) step
      in
      let init_uses = uses st inits in
      let res = results st rs in
      let ps = params st body in
      line st indent "%sfor %s iter(%s := %s) {" res header_uses ps init_uses;
      region st (indent + 1) body;
      line st indent "}"
  | Ssa_stmt.If { cond; results = rs; then_; else_ } ->
      let c = name st cond in
      let res = results st rs in
      line st indent "%sif %s {" res c;
      region st (indent + 1) then_;
      line st indent "} else {";
      region st (indent + 1) else_;
      line st indent "}"
  | Ssa_stmt.Instr i ->
      let text = op st i.Ssa_instr.op in
      let token =
        match i.Ssa_instr.token with
        | None -> ""
        | Some e -> Fmt.str " effect %s" (name st e)
      in
      let origin =
        match i.Ssa_instr.origin with
        | Ssa_origin.Unknown -> ""
        | o -> Fmt.str "  ; %a" Ssa_origin.pp o
      in
      let res = results st i.Ssa_instr.results in
      line st indent "%s%s%s%s" res text token origin
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token; results = rs; body } ->
      let header = Fmt.str "%s to %s" (name st lo) (name st hi) in
      let init = Fmt.str "%s, %s" (name st seed) (name st token) in
      let res = results st rs in
      let ps = params st body in
      line st indent "%sordered_sum %s iter(%s := %s) {" res header ps init;
      region st (indent + 1) body;
      line st indent "}"

let buffer st (b : Ssa_buffer.t) =
  line st 0 "buffer %a %s %s [%s]" Ssa_id.Buffer.pp b.Ssa_buffer.id
    (Ssa_buffer.role_name b.Ssa_buffer.role)
    (Ssa_format.name b.Ssa_buffer.format)
    (join (List.map Int64.to_string (Expr.Coord.to_list b.Ssa_buffer.extents)))

let to_string (p : Ssa_program.t) =
  let st = { out = Buffer.create 256; names = Hashtbl.create 64; next = 0 } in
  List.iter (buffer st) p.Ssa_program.buffers;
  let entry = p.Ssa_program.entry in
  line st 0 "entry(%s) {" (params st entry);
  region st 1 entry;
  line st 0 "}";
  Buffer.contents st.out

let pp fmt p = Fmt.string fmt (to_string p)
