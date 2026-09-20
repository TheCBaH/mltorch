(* A run of Native reshape, clone and permute steps between two four-axis
   tensors is one index permutation of the flat data, however many axes its
   interior tensors carry. This module turns such a run into at most a few
   [Reshape4]/[Permute4] steps over the four-axis frame.

   The data is split into ATOMS: the pieces of each axis that no reshape in the
   run cuts across. Every step moves whole atoms, so the run is a permutation
   of the atoms of the source. Atoms that stay adjacent, in the same order, in
   both source and result fuse into one block; the blocks are then permuted by
   a shortest sequence of steps, each of which cuts the current order into at
   most four contiguous runs and reorders them (what one [Permute4] can do).

   [None] means "outside this planner" -- a reshape whose cut falls inside an
   atom, more blocks than [max_blocks], or no order reachable within
   [max_permutes] steps. The caller then leaves the run to the ordinary path,
   which names the real blocker. *)

type op =
  | Clone
  | Permute of Permute.Permute.perm
  | Reshape of Vec6.shape
      (** One Native step, with a reshape carrying its target shape. *)

type step =
  | Reshape4 of Shape4.t
  | Permute4 of (Axis4.t * Axis4.t) list * Shape4.t
      (** The permutation, and the shape it produces. *)

val max_blocks : int
val max_permutes : int

val plan :
  ?source:Vec6.shape -> x:Shape4.t -> y:Shape4.t -> op list -> step list option
(** Steps taking a tensor of shape [x] to one of shape [y] through the given
    ops, in order; [[]] when neither the data nor the shape changes. The atoms
    are those of [source] when given (the tensor's own six-axis shape, of which
    [x] is a four-axis reading), else of [x]. *)
