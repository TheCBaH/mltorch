(* Where an instruction came from. One CFG operation may expand into several
   instructions; each keeps the operation's location, its role in the expansion
   and a clone instance. Unknown stays unknown: no location is invented. *)

module Cfg_site = struct
  (* The CFG's own block ids, which are not [Mir_id.Block]'s. *)
  module Block =
    Core.Tagged_int.Make
      (struct
        let prefix = "bb"
      end)
      ()

  (* An operation's position in its CFG block's body. *)
  module Instr =
    Core.Tagged_int.Make
      (struct
        let prefix = ""
      end)
      ()

  type position = Instr of Instr.t | Terminator

  (* The CFG block id and the operation's position in its body, or the block's
     terminator. *)
  type t = { block : Block.t; position : position }

  let equal (a : t) b = a = b
end

(* Which copy of a cloned CFG operation an instruction belongs to. *)
module Clone =
  Core.Tagged_int.Make
    (struct
      let prefix = "clone"
    end)
    ()

module Role = struct
  type t =
    | Address  (** byte-offset and pointer formation *)
    | Compute  (** the operation's own value *)
    | Decode  (** storage-format decode after a raw load *)
    | Encode  (** storage-format encode before a raw store *)
    | Guard  (** a source-failure test and its branch *)
    | Payload  (** a failure payload value *)
    | Transfer  (** an edge copy or block argument *)

  let name = function
    | Address -> "address"
    | Compute -> "compute"
    | Decode -> "decode"
    | Encode -> "encode"
    | Guard -> "guard"
    | Payload -> "payload"
    | Transfer -> "transfer"
end

type t = {
  cfg : Cfg_site.t option;
  output : Expr.Source.t option;  (** the output the SSA origin names *)
  role : Role.t;
  clone : Clone.t;
}

let unknown =
  { cfg = None; output = None; role = Role.Compute; clone = Clone.of_int 0 }

let pp fmt t =
  match t.cfg with
  | None -> Fmt.string fmt "?"
  | Some { Cfg_site.block; position } ->
      Fmt.pf fmt "cfg %a.%a %s" Cfg_site.Block.pp block
        (fun fmt -> function
          | Cfg_site.Instr i -> Cfg_site.Instr.pp fmt i
          | Cfg_site.Terminator -> Fmt.string fmt "term")
        position (Role.name t.role)
