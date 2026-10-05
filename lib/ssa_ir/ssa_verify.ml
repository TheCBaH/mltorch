module Statement =
  Core.Tagged_int.Make
    (struct
      let prefix = "stmt"
    end)
    ()

type site = { region : Ssa_id.Region.t; statement : Statement.t }

type problem =
  | Buffer_declaration of Ssa_id.Buffer.t
  | Buffer_format of {
      buffer : Ssa_id.Buffer.t;
      accessed : Ssa_format.t;
      declared : Ssa_format.t;
    }
  | Buffer_not_stored of Ssa_id.Buffer.t
  | Buffer_unknown of Ssa_id.Buffer.t
  | Definition_twice of Ssa_id.Value.t
  | Effect_missing
  | Effect_stale of { used : Ssa_value.t; live : Ssa_value.t }
  | Effect_unique of Ssa_type.t list
  | Region_defined_twice of Ssa_id.Region.t
  | Region_too_deep
  | Result_types of { declared : Ssa_type.t list; expected : Ssa_type.t list }
  | Signature of { expected : Ssa_type.t list; found : Ssa_type.t list }
  | Step_not_positive of int64
  | Typing of Ssa_typing.error
  | Use_retyped of {
      value : Ssa_id.Value.t;
      defined : Ssa_type.t;
      used : Ssa_type.t;
    }
  | Use_undefined of Ssa_id.Value.t

type diagnostic = { site : site; problem : problem }
type error = [ `Invalid_program of diagnostic ]

let pp_types = Fmt.list ~sep:(Fmt.any ", ") Ssa_type.pp

let pp_problem fmt = function
  | Buffer_declaration b ->
      Fmt.pf fmt "buffer %a is declared twice, or with extents no index holds"
        Ssa_id.Buffer.pp b
  | Buffer_format { buffer; accessed; declared } ->
      Fmt.pf fmt "%a is declared %s but accessed as %s" Ssa_id.Buffer.pp buffer
        (Ssa_format.name declared) (Ssa_format.name accessed)
  | Buffer_not_stored b ->
      Fmt.pf fmt "%a is an input and is never written" Ssa_id.Buffer.pp b
  | Buffer_unknown b -> Fmt.pf fmt "%a is not declared" Ssa_id.Buffer.pp b
  | Definition_twice v -> Fmt.pf fmt "%a is defined twice" Ssa_id.Value.pp v
  | Effect_missing -> Fmt.string fmt "effect operand missing or unexpected"
  | Effect_stale { used; live } ->
      Fmt.pf fmt "effect %a is not the current effect %a" Ssa_id.Value.pp
        used.Ssa_value.id Ssa_id.Value.pp live.Ssa_value.id
  | Effect_unique ts ->
      Fmt.pf fmt "a region carries exactly one effect, signature (%a)" pp_types
        ts
  | Region_defined_twice r ->
      Fmt.pf fmt "region %a appears twice" Ssa_id.Region.pp r
  | Region_too_deep -> Fmt.string fmt "region nesting is too deep"
  | Result_types { declared; expected } ->
      Fmt.pf fmt "results declared (%a), operation yields (%a)" pp_types
        declared pp_types expected
  | Signature { expected; found } ->
      Fmt.pf fmt "signature (%a) expected, (%a) found" pp_types expected
        pp_types found
  | Step_not_positive s -> Fmt.pf fmt "loop step %Ld is not positive" s
  | Typing e -> Ssa_typing.pp_error fmt e
  | Use_retyped { value; defined; used } ->
      Fmt.pf fmt "%a is defined as %a and used as %a" Ssa_id.Value.pp value
        Ssa_type.pp defined Ssa_type.pp used
  | Use_undefined v ->
      Fmt.pf fmt "%a is not in scope at its use" Ssa_id.Value.pp v

let pp_diagnostic fmt { site; problem } =
  Fmt.pf fmt "%a, %a: %a" Ssa_id.Region.pp site.region Statement.pp
    site.statement pp_problem problem

let pp_error fmt : [< error ] -> unit = function
  | `Invalid_program d -> pp_diagnostic fmt d

let max_region_depth = 256

type ctx = {
  esc : error Err.Escape.t;
  program : Ssa_program.t;
  mutable defined : Ssa_id.Value.Set.t;
  mutable regions : Ssa_id.Region.Set.t;
}

type scope = {
  region : Ssa_id.Region.t;
  statement : Statement.t;
  env : Ssa_type.t Ssa_id.Value.Map.t;
}

let fail ctx (scope : scope) problem =
  Err.Escape.throw ctx.esc
    (`Invalid_program
       {
         site = { region = scope.region; statement = scope.statement };
         problem;
       }
      : error)

let define ctx scope (v : Ssa_value.t) =
  if Ssa_id.Value.Set.mem v.Ssa_value.id ctx.defined then
    fail ctx scope (Definition_twice v.Ssa_value.id);
  ctx.defined <- Ssa_id.Value.Set.add v.Ssa_value.id ctx.defined;
  {
    scope with
    env = Ssa_id.Value.Map.add v.Ssa_value.id v.Ssa_value.ty scope.env;
  }

let define_all ctx scope vs = List.fold_left (define ctx) scope vs

let use ctx scope (v : Ssa_value.t) =
  match Ssa_id.Value.Map.find_opt v.Ssa_value.id scope.env with
  | None -> fail ctx scope (Use_undefined v.Ssa_value.id)
  | Some ty ->
      if not (Ssa_type.equal ty v.Ssa_value.ty) then
        fail ctx scope
          (Use_retyped
             { value = v.Ssa_value.id; defined = ty; used = v.Ssa_value.ty })

let types vs = List.map (fun (v : Ssa_value.t) -> v.Ssa_value.ty) vs
let is_effect (v : Ssa_value.t) = Ssa_type.equal v.Ssa_value.ty Ssa_type.Effect

let expect_type ctx scope expected (v : Ssa_value.t) =
  use ctx scope v;
  if not (Ssa_type.equal v.Ssa_value.ty expected) then
    fail ctx scope
      (Typing
         (Ssa_typing.Operand
            {
              position = Ssa_typing.Position.of_int 0;
              expected;
              found = v.Ssa_value.ty;
            }))

let expect_signature ctx scope ~expected ~found =
  if
    not
      (List.length expected = List.length found
      && List.for_all2 Ssa_type.equal expected found)
  then fail ctx scope (Signature { expected; found })

(* The effect-carrying values of a signature: exactly one. *)
let effect_of ctx scope vs =
  match List.filter is_effect vs with
  | [ e ] -> e
  | _ -> fail ctx scope (Effect_unique (types vs))

let consume ctx scope ~live (e : Ssa_value.t) =
  use ctx scope e;
  if not (Ssa_value.equal e live) then
    fail ctx scope (Effect_stale { used = e; live })

let buffer ctx scope id =
  match Ssa_program.find_buffer ctx.program id with
  | Some b -> b
  | None -> fail ctx scope (Buffer_unknown id)

let check_access ctx scope id ~format ~writes =
  let b = buffer ctx scope id in
  if b.Ssa_buffer.format <> format then
    fail ctx scope
      (Buffer_format
         { buffer = id; accessed = format; declared = b.Ssa_buffer.format });
  if writes && b.Ssa_buffer.role = Ssa_buffer.Input then
    fail ctx scope (Buffer_not_stored id)

let instr ctx scope ~live (i : Ssa_instr.t) =
  let op = i.Ssa_instr.op in
  List.iter (use ctx scope) (Ssa_op.operands op);
  (match op with
  | Ssa_op.Load { buffer; decode; _ } ->
      check_access ctx scope buffer
        ~format:(Ssa_op.Decode.format decode)
        ~writes:false
  | Ssa_op.Store { buffer; encode; _ } ->
      check_access ctx scope buffer
        ~format:(Ssa_op.Encode.format encode)
        ~writes:true
  | Ssa_op.Const _ | Ssa_op.Convert _ | Ssa_op.Float_binary _
  | Ssa_op.Index_add _ | Ssa_op.Index_scale _ | Ssa_op.Mark _ ->
      ());
  let values =
    match Ssa_typing.result_types op with
    | Ok ts -> ts
    | Error e -> fail ctx scope (Typing e)
  in
  let expected =
    if Ssa_op.effectful op then values @ [ Ssa_type.Effect ] else values
  in
  let declared = types i.Ssa_instr.results in
  if
    not
      (List.length declared = List.length expected
      && List.for_all2 Ssa_type.equal declared expected)
  then fail ctx scope (Result_types { declared; expected });
  let live =
    match (Ssa_op.effectful op, i.Ssa_instr.token) with
    | true, Some e ->
        consume ctx scope ~live e;
        List.nth i.Ssa_instr.results (List.length i.Ssa_instr.results - 1)
    | false, None -> live
    | true, None | false, Some _ -> fail ctx scope Effect_missing
  in
  (define_all ctx scope i.Ssa_instr.results, live)

let rec region ctx scope ~depth ~live_of (r : Ssa_region.t) =
  if depth > max_region_depth then fail ctx scope Region_too_deep;
  if Ssa_id.Region.Set.mem r.Ssa_region.id ctx.regions then
    fail ctx scope (Region_defined_twice r.Ssa_region.id);
  ctx.regions <- Ssa_id.Region.Set.add r.Ssa_region.id ctx.regions;
  let scope =
    { scope with region = r.Ssa_region.id; statement = Statement.of_int 0 }
  in
  let scope = define_all ctx scope r.Ssa_region.params in
  let live = live_of scope r.Ssa_region.params in
  let scope, live =
    List.fold_left
      (fun ((scope : scope), live) s ->
        let scope', live = stmt ctx scope ~depth ~live s in
        ( {
            scope' with
            statement = Statement.of_int (Statement.to_int scope.statement + 1);
          },
          live ))
      (scope, live) r.Ssa_region.body
  in
  List.iter (use ctx scope) r.Ssa_region.yields;
  (match List.filter is_effect r.Ssa_region.yields with
  | [ e ] -> consume ctx scope ~live e
  | _ -> fail ctx scope (Effect_unique (types r.Ssa_region.yields)));
  types r.Ssa_region.yields

and stmt ctx scope ~depth ~live : Ssa_region.t Ssa_stmt.t -> scope * Ssa_value.t
    = function
  | Ssa_stmt.For { lo; hi; step; inits; results; body } ->
      let index = Ssa_type.Scalar Ssa_type.Index in
      expect_type ctx scope index lo;
      expect_type ctx scope index hi;
      if Int64.compare step 0L <= 0 then fail ctx scope (Step_not_positive step);
      List.iter (use ctx scope) inits;
      consume ctx scope ~live (effect_of ctx scope inits);
      (match body.Ssa_region.params with
      | iv :: carried ->
          expect_signature ctx scope ~expected:[ index ]
            ~found:[ iv.Ssa_value.ty ];
          expect_signature ctx scope ~expected:(types inits)
            ~found:(types carried)
      | [] -> fail ctx scope (Signature { expected = [ index ]; found = [] }));
      let yielded =
        region ctx scope ~depth:(depth + 1)
          ~live_of:(fun scope params ->
            effect_of ctx scope (match params with _ :: c -> c | [] -> []))
          body
      in
      expect_signature ctx scope ~expected:(types inits) ~found:yielded;
      expect_signature ctx scope ~expected:yielded ~found:(types results);
      let scope = define_all ctx scope results in
      (scope, effect_of ctx scope results)
  | Ssa_stmt.If { cond; results; then_; else_ } ->
      expect_type ctx scope (Ssa_type.Scalar Ssa_type.Pred) cond;
      let branch r =
        let yielded =
          region ctx scope ~depth:(depth + 1)
            ~live_of:(fun scope params ->
              if params <> [] then
                fail ctx scope
                  (Signature { expected = []; found = types params });
              live)
            r
        in
        expect_signature ctx scope ~expected:(types results) ~found:yielded
      in
      branch then_;
      branch else_;
      let scope = define_all ctx scope results in
      (scope, effect_of ctx scope results)
  | Ssa_stmt.Instr i -> instr ctx scope ~live i
  | Ssa_stmt.Ordered_sum { lo; hi; seed; token; results; body } ->
      let index = Ssa_type.Scalar Ssa_type.Index in
      expect_type ctx scope index lo;
      expect_type ctx scope index hi;
      use ctx scope seed;
      (match seed.Ssa_value.ty with
      | Ssa_type.Scalar (Ssa_type.F32 | Ssa_type.F64) -> ()
      | found ->
          fail ctx scope
            (Typing
               (Ssa_typing.Operand_not_float
                  { position = Ssa_typing.Position.of_int 0; found })));
      consume ctx scope ~live token;
      let tys = [ seed.Ssa_value.ty; Ssa_type.Effect ] in
      expect_signature ctx scope ~expected:tys ~found:(types results);
      (match body.Ssa_region.params with
      | [ iv; e ] ->
          expect_signature ctx scope ~expected:[ index; Ssa_type.Effect ]
            ~found:[ iv.Ssa_value.ty; e.Ssa_value.ty ]
      | params ->
          fail ctx scope
            (Signature
               { expected = [ index; Ssa_type.Effect ]; found = types params }));
      let yielded =
        region ctx scope ~depth:(depth + 1)
          ~live_of:(fun _ params ->
            match params with [ _; e ] -> e | _ -> token)
          body
      in
      expect_signature ctx scope ~expected:tys ~found:yielded;
      let scope = define_all ctx scope results in
      (scope, effect_of ctx scope results)

let check_buffers ctx scope =
  let seen = ref Ssa_id.Buffer.Set.empty in
  List.iter
    (fun (b : Ssa_buffer.t) ->
      if
        Ssa_id.Buffer.Set.mem b.Ssa_buffer.id !seen
        || Ssa_buffer.elements b.Ssa_buffer.extents = None
      then fail ctx scope (Buffer_declaration b.Ssa_buffer.id);
      seen := Ssa_id.Buffer.Set.add b.Ssa_buffer.id !seen)
    ctx.program.Ssa_program.buffers

let check (p : Ssa_program.t) =
  Err.Escape.with_escape @@ fun esc ->
  let ctx =
    {
      esc;
      program = p;
      defined = Ssa_id.Value.Set.empty;
      regions = Ssa_id.Region.Set.empty;
    }
  in
  let scope =
    {
      region = p.Ssa_program.entry.Ssa_region.id;
      statement = Statement.of_int 0;
      env = Ssa_id.Value.Map.empty;
    }
  in
  check_buffers ctx scope;
  let yielded =
    region ctx scope ~depth:0
      ~live_of:(fun scope params -> effect_of ctx scope params)
      p.Ssa_program.entry
  in
  expect_signature ctx scope ~expected:[ Ssa_type.Effect ] ~found:yielded
