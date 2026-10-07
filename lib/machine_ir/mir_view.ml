(* A window of a region with a permission and a role: what a buffer descriptor
   or a pointer parameter names. [source] is the tensor or buffer identity a
   failure row reports, never an alias proof. *)

type perm = Read | Read_write | Write

type role =
  | Constant
  | Input
  | Output
  | Runtime  (** invocation state: the scan meter, failure record *)
  | Scratch

type t = {
  id : Mir_id.View.t;
  region : Mir_id.Region.t;
  offset : int64;
  size : int64;
  perm : perm;
  role : role;
  source : Expr.Source.t option;
}

let readable = function Read | Read_write -> true | Write -> false
let writable = function Write | Read_write -> true | Read -> false
let perm_name = function Read -> "r" | Read_write -> "rw" | Write -> "w"

let role_name = function
  | Constant -> "const"
  | Input -> "in"
  | Output -> "out"
  | Runtime -> "runtime"
  | Scratch -> "scratch"
