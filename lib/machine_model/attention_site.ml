open Ssa_ir

type t = {
  block : Attention_block.t;
  query : Tensor_id.t;
  key : Tensor_id.t;
  value : Tensor_id.t;
  mask : Tensor_id.t option;
  output : Tensor_id.t;
}

module Buffers = struct
  type t = {
    q : Ssa_id.Buffer.t;
    k : Ssa_id.Buffer.t;
    v : Ssa_id.Buffer.t;
    mask : Ssa_id.Buffer.t option;
    o : Ssa_id.Buffer.t;
  }

  let pp ppf t =
    Fmt.pf ppf "q=%a k=%a v=%a mask=%a o=%a" Ssa_id.Buffer.pp t.q
      Ssa_id.Buffer.pp t.k Ssa_id.Buffer.pp t.v
      Fmt.(option ~none:(any "none") Ssa_id.Buffer.pp)
      t.mask Ssa_id.Buffer.pp t.o
end

module Unbound = struct
  type t =
    | Not_an_output of Tensor_id.t
    | Role of { role : string; tensor : Tensor_id.t }

  let pp ppf = function
    | Not_an_output tensor ->
        Fmt.pf ppf "the attention's output %a is not one the program writes"
          Tensor_id.pp tensor
    | Role { role; tensor } ->
        Fmt.pf ppf "the attention's %s %a is bound by no buffer" role
          Tensor_id.pp tensor
end

let buffers (t : t) (inv : Loop_ir.Loop_bundle.invocation) =
  let positional =
    List.combine inv.Loop_ir.Loop_bundle.program.Loop_ir.Loop_program.buffers
      inv.Loop_ir.Loop_bundle.edges
  in
  let find role tensor =
    match
      List.find_opt (fun (_, edge) -> Tensor_id.equal edge tensor) positional
    with
    | Some (b, _) ->
        Ok (Ssa_id.Buffer.of_int (Tensor_id.to_int b.Loop_ir.Loop_buffer.id), b)
    | None -> Error (Unbound.Role { role; tensor })
  in
  let ( let* ) = Result.bind in
  let* q, _ = find "query" t.query in
  let* k, _ = find "key" t.key in
  let* v, _ = find "value" t.value in
  let* mask =
    match t.mask with
    | None -> Ok None
    | Some m ->
        let* id, _ = find "mask" m in
        Ok (Some id)
  in
  let* o, ob = find "output" t.output in
  match ob.Loop_ir.Loop_buffer.role with
  | Loop_ir.Loop_buffer.Output -> Ok { Buffers.q; k; v; mask; o }
  | _ -> Error (Unbound.Not_an_output t.output)

let program (t : t) (b : Buffers.t) (p : Ssa_program.t) =
  match p.Ssa_program.entry.Ssa_region.params with
  | [ effect ] when effect.Ssa_value.ty = Ssa_type.Effect ->
      let id, next = Ssa_id.Value.Next.alloc p.Ssa_program.next_value in
      let after = { Ssa_value.id; ty = Ssa_type.Effect } in
      let instr =
        Ssa_stmt.Instr
          {
            Ssa_instr.results = [ after ];
            op =
              Ssa_op.Attend
                {
                  desc = t.block;
                  q = b.Buffers.q;
                  k = b.Buffers.k;
                  v = b.Buffers.v;
                  mask = b.Buffers.mask;
                  o = b.Buffers.o;
                };
            token = Some effect;
            origin = Ssa_origin.Unknown;
          }
      in
      Ok
        {
          p with
          Ssa_program.revision = Ssa_id.Revision.succ p.Ssa_program.revision;
          entry =
            {
              p.Ssa_program.entry with
              Ssa_region.body = [ instr ];
              yields = [ after ];
            };
          next_value = next;
        }
  | _ -> Error "the program's entry does not take exactly one effect"
