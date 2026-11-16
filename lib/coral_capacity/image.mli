(** What a segment needs of a configuration's memory: its executable size and
    every data allocation, in the order the layout places them. *)

module Allocation : sig
  type t = {
    align : Data_bytes.t;
    bytes : Data_bytes.t;
    name : string;
    placement : Coral_layout.Placement.t;
  }
end

type t = { allocations : Allocation.t list; code : Code_bytes.t; name : string }
