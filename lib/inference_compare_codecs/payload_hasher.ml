open Inference_compare_schema

module type BACKEND = sig
  type state

  val algorithm : Id.Digest_algorithm.t
  val create : unit -> state
  val update : state -> string -> pos:int -> len:int -> unit
  val finish : state -> string
end

module Registry = struct
  type t = (module BACKEND) list

  let empty = []

  let register t (module B : BACKEND) =
    let exists (module X : BACKEND) =
      Id.Digest_algorithm.equal X.algorithm B.algorithm
    in
    if List.exists exists t then Err.fail (`Duplicate_algorithm B.algorithm)
    else Ok (t @ [ (module B : BACKEND) ])

  let algorithms t = List.map (fun (module X : BACKEND) -> X.algorithm) t

  let find t algorithm =
    List.find_opt
      (fun (module X : BACKEND) ->
        Id.Digest_algorithm.equal X.algorithm algorithm)
      t
end

type error =
  | Finished
  | Invalid_range of { length : int; pos : int; len : int }
  | Unknown_algorithm of Id.Digest_algorithm.t

let pp_error ppf = function
  | Finished -> Fmt.string ppf "digest state already finished"
  | Invalid_range { length; pos; len } ->
      Fmt.pf ppf "range pos=%d len=%d outside length %d" pos len length
  | Unknown_algorithm a ->
      Fmt.pf ppf "unknown digest algorithm %a" Id.Digest_algorithm.pp a

module State = struct
  type t = {
    algorithm : Id.Digest_algorithm.t;
    update : string -> pos:int -> len:int -> unit;
    finish : unit -> string;
    mutable finished : bool;
  }
end

let fail e = Err.fail ~pp_error e

let create registry algorithm =
  match Registry.find registry algorithm with
  | None -> fail (Unknown_algorithm algorithm)
  | Some (module B : BACKEND) ->
      let st = B.create () in
      Ok
        {
          State.algorithm;
          update = (fun s ~pos ~len -> B.update st s ~pos ~len);
          finish = (fun () -> B.finish st);
          finished = false;
        }

let update (st : State.t) s ~pos ~len =
  if st.finished then fail Finished
  else if pos < 0 || len < 0 || pos > String.length s - len then
    fail (Invalid_range { length = String.length s; pos; len })
  else (
    st.update s ~pos ~len;
    Ok ())

let update_string st s = update st s ~pos:0 ~len:(String.length s)

let finish (st : State.t) =
  if st.finished then fail Finished
  else (
    st.finished <- true;
    let hex = st.finish () in
    match Digest.Payload.make st.algorithm ~hex with
    | Ok d -> Ok d
    | Error _ ->
        (* A backend that returns non-canonical hex is a registration bug, but
           it must surface as an error, not as a malformed digest. *)
        fail (Unknown_algorithm st.algorithm))

let digest_string registry algorithm s =
  let ( let* ) = Err.bind in
  let* st = create registry algorithm in
  let* () = update_string st s in
  finish st

let content_path d =
  Id.Digest_algorithm.to_string (Digest.Payload.algorithm d)
  ^ "/" ^ Digest.Payload.hex d
