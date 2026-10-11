(** The protocol between the comparison driver and the JavaScript engine helper
    (version {!version}).

    A message is a 4-byte little-endian length and a body: the magic ["MLCJ"],
    the protocol version as 4 bytes, a kind byte and the kind's fields. A string
    is a 4-byte length and its bytes, a tensor its format tag, six 4-byte
    extents, an 8-byte byte count and the row-major little-endian cell bytes,
    exactly as the engine holds them. A message of another version, or one that
    is truncated or has trailing bytes, is rejected whole: nothing is guessed.

    The driver sends {!request}s one at a time and reads one {!response} to
    each. A body is prepared once under a handle the driver chooses and may be
    run any number of times before it is released. *)

val version : int

type request =
  | Hello
  | Prepare of { handle : int; engine : string; graph : string }
      (** [graph] is {!Graph_ir.graph_jsont}'s JSON. *)
  | Run of { handle : int; inputs : (int * Tensor.packed) list }
      (** Inputs by tensor id. *)
  | Release of int

type response =
  | Welcome of { engines : string list; runtime : string }
  | Prepared
  | Refused of string  (** the engine declined the body when preparing it *)
  | Outputs of Tensor.packed list  (** in the graph's output order *)
  | Failed of string  (** the engine failed while running *)
  | Released

val encode_request : request -> string
val encode_response : response -> string

val decode_request : string -> (request, string) result
(** From a message body, without its length prefix. *)

val decode_response : string -> (response, string) result

val frame : string -> string
(** The body with its length prefix. *)

val frame_length : string -> (int, string) result
(** The body length a 4-byte prefix announces, refused if above {!max_body}. *)

val max_body : int
(** The largest body either side accepts. *)
