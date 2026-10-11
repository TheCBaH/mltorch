(** What the two compiled routes are made of, and which parts they share. *)

type component = {
  name : string;
  pin : string;  (** The pinned source revision or artifact digest. *)
  routes : [ `Both | `Compcert | `Assembly ];
}

val components : component list

val shared : component list
(** The parts both routes end in. An agreement between the routes does not vouch
    for these. *)
