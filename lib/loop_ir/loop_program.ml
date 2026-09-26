type t = {
  buffers : Loop_buffer.t list;
  body : Loop_stmt.t list;
  scan_limits : Expr.Scan_limits.t;
      (** The budget a scan meter charges against. *)
  max_depth : int;
      (** The [Kernel.Limits.max_depth] every lowered expression was checked
          under: the bound the interpreter's expression recursion relies on. *)
}
