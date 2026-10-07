(* The order state an effectful instruction consumes and the state it
   produces. Order values form one chain per path: a block starts at its order
   parameter and every effectful instruction, edge, return and failure exit
   consumes the latest state. *)
type t = { input : Mir_value.t; output : Mir_value.t }
