module Allocation = struct
  type t = {
    align : Data_bytes.t;
    bytes : Data_bytes.t;
    name : string;
    placement : Coral_layout.Placement.t;
  }
end

type t = { allocations : Allocation.t list; code : Code_bytes.t; name : string }
