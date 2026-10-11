module Ledger = struct
  type t = {
    limit : int64 option;
    mutable blobs : int;
    mutable bytes : int64;
    mutable peak : int64;
    mutable leases : int;
  }

  let create ?limit () = { limit; blobs = 0; bytes = 0L; peak = 0L; leases = 0 }
  let live_blobs t = t.blobs
  let live_bytes t = t.bytes
  let peak_bytes t = t.peak
  let outstanding_leases t = t.leases
  let balanced t = t.blobs = 0 && t.leases = 0
end

type t = {
  ledger : Ledger.t;
  data : string;
  mutable holders : int;
  mutable dropped : bool;
}

let length t = Int64.of_int (String.length t.data)
let alive t = t.holders > 0

let contents t =
  if alive t then t.data else invalid_arg "Owned_blob.contents: released"

let create (ledger : Ledger.t) data =
  let size = Int64.of_int (String.length data) in
  (* Compare against the headroom rather than adding: [bytes + size] could
     wrap, whereas [limit - bytes] cannot go negative while bytes <= limit. *)
  let over =
    match ledger.limit with
    | None -> false
    | Some limit -> Int64.compare size (Int64.sub limit ledger.bytes) > 0
  in
  if over then
    Err.fail ~pp_error:Codec_error.pp
      (Codec_error.Resource_exceeded
         {
           requested = size;
           limit = Option.value ledger.limit ~default:Int64.max_int;
         })
  else begin
    ledger.blobs <- ledger.blobs + 1;
    ledger.bytes <- Int64.add ledger.bytes size;
    if Int64.compare ledger.bytes ledger.peak > 0 then
      ledger.peak <- ledger.bytes;
    Ok { ledger; data; holders = 1; dropped = false }
  end

let unhold t =
  t.holders <- t.holders - 1;
  if t.holders = 0 then begin
    t.ledger.blobs <- t.ledger.blobs - 1;
    t.ledger.bytes <- Int64.sub t.ledger.bytes (length t)
  end

let drop t =
  if t.dropped then invalid_arg "Owned_blob.drop: already dropped";
  t.dropped <- true;
  unhold t

module Lease = struct
  type blob = t
  type t = { blob : blob; mutable released : bool }

  let release l =
    if l.released then invalid_arg "Owned_blob.Lease.release: already released";
    l.released <- true;
    l.blob.ledger.leases <- l.blob.ledger.leases - 1;
    unhold l.blob

  let contents l =
    if l.released then invalid_arg "Owned_blob.Lease.contents: released";
    l.blob.data

  let blob l = l.blob
end

let lease t =
  if not (alive t) then invalid_arg "Owned_blob.lease: released";
  t.holders <- t.holders + 1;
  t.ledger.leases <- t.ledger.leases + 1;
  { Lease.blob = t; released = false }
