(* Synthetic byte memory. Each region instance has a deterministic synthetic
   base address, sparse 4 KiB pages of bytes and a per-byte initialization
   map; nothing here is a host pointer. A pointer keeps its region instance,
   the window of the view it came from and its offset in the region, so an
   access checks the complete byte range, wraparound, alignment, permission and
   initialization. One-past (and further) pointers may exist; only an access
   dereferences. *)

open Machine_ir

let page_bits = 12
let page_size = 1 lsl page_bits

module Page = struct
  type t = { data : Bytes.t; init : Bytes.t  (** 1 per defined byte *) }

  let create () =
    { data = Bytes.make page_size '\000'; init = Bytes.make page_size '\000' }
end

(* A region instance within one memory: its own domain, never a region id. *)
module Key =
  Core.Tagged_int.Make
    (struct
      let prefix = "inst"
    end)
    ()

module Instance = struct
  type t = {
    key : Key.t;  (** this memory's own instance number *)
    region : Mir_id.Region.t option;  (** [None] for a fresh dynamic object *)
    base : int64;  (** synthetic address *)
    size : int64;
    pages : (int64, Page.t) Hashtbl.t;
    mutable live : bool;
  }
end

module Pointer = struct
  type t = {
    instance : Key.t;
    lo : int64;  (** the view window's first byte, as a region offset *)
    hi : int64;  (** one past its last byte *)
    perm : Mir_view.perm;
    offset : int64;
  }

  let equal (a : t) b = a = b
end

type t = {
  instances : (Key.t, Instance.t) Hashtbl.t;
  mutable next_key : Key.Next.t;
  mutable next_base : int64;
}

(* The first synthetic base, and the unmapped gap after every instance so an
   address one past an instance never names the next one. *)
let first_base = 0x1000_0000L
let gap = 0x1_0000L

let create () =
  {
    instances = Hashtbl.create 16;
    next_key = Key.Next.first;
    next_base = first_base;
  }

let instance t key =
  match Hashtbl.find_opt t.instances key with
  | Some i -> i
  | None -> invalid_arg "Mir_memory: no such instance"

(* An instance's size in bytes. *)
let size t key = (instance t key).Instance.size
let align_up x a = Option.get (Mir_layout.align_up x ~align:a)

(* A new instance of [size] bytes at a fresh base aligned to [align] (at least
   a page). [None] if the synthetic address space is exhausted. *)
let alloc t ?region ~size ~align () =
  let align = if Int64.compare align 4096L < 0 then 4096L else align in
  match Mir_layout.align_up t.next_base ~align with
  | None -> None
  | Some base -> (
      match Mir_layout.add base size with
      | None -> None
      | Some last -> (
          match Mir_layout.add last gap with
          | None -> None
          | Some next ->
              Key.Next.check_room t.next_key ~count:1;
              let key, next_key = Key.Next.alloc t.next_key in
              t.next_key <- next_key;
              t.next_base <- align_up next 4096L;
              Hashtbl.replace t.instances key
                {
                  Instance.key;
                  region;
                  base;
                  size;
                  pages = Hashtbl.create 4;
                  live = true;
                };
              Some key))

let page (i : Instance.t) n =
  match Hashtbl.find_opt i.Instance.pages n with
  | Some p -> p
  | None ->
      let p = Page.create () in
      Hashtbl.replace i.Instance.pages n p;
      p

let locate off =
  ( Int64.shift_right_logical off page_bits,
    Int64.to_int (Int64.logand off (Int64.of_int (page_size - 1))) )

let set_byte i off b =
  let n, k = locate off in
  let p = page i n in
  Bytes.set p.Page.data k (Char.unsafe_chr (b land 0xFF));
  Bytes.set p.Page.init k '\001'

(* [None] for an undefined byte. *)
let get_byte (i : Instance.t) off =
  let n, k = locate off in
  match Hashtbl.find_opt i.Instance.pages n with
  | None -> None
  | Some p ->
      if Bytes.get p.Page.init k = '\000' then None
      else Some (Char.code (Bytes.get p.Page.data k))

(* Every byte of the pointer's view window undefined again: a fresh logical
   lifetime. *)
let undefine t (p : Pointer.t) =
  let i = instance t p.Pointer.instance in
  let rec go off =
    if Int64.compare off p.Pointer.hi < 0 then (
      let n, k = locate off in
      let next = Int64.shift_left (Int64.succ n) page_bits in
      let stop =
        if Int64.compare next p.Pointer.hi < 0 then next else p.Pointer.hi
      in
      (match Hashtbl.find_opt i.Instance.pages n with
      | Some page ->
          Bytes.fill page.Page.init k (Int64.to_int (Int64.sub stop off)) '\000'
      | None -> ());
      go stop)
  in
  go p.Pointer.lo

let write_string t key ~offset s =
  let i = instance t key in
  String.iteri
    (fun k c -> set_byte i (Int64.add offset (Int64.of_int k)) (Char.code c))
    s

module Fault = struct
  type t = Bad_access | Uninitialized
end

let pointer ?(perm = Mir_view.Read_write) t key ~lo ~hi =
  ignore (instance t key);
  { Pointer.instance = key; lo; hi; perm; offset = lo }

let address t (p : Pointer.t) =
  let i = instance t p.Pointer.instance in
  Int64.add i.Instance.base p.Pointer.offset

(* [p + delta]; [None] if the region offset would wrap. *)
let offset_by (p : Pointer.t) delta =
  let o = p.Pointer.offset in
  let r = Int64.add o delta in
  let wrapped =
    (Int64.compare delta 0L > 0 && Int64.compare r o < 0)
    || (Int64.compare delta 0L < 0 && Int64.compare r o > 0)
  in
  if wrapped then None else Some { p with Pointer.offset = r }

let check t (p : Pointer.t) ~bytes ~align ~write =
  let i = instance t p.Pointer.instance in
  let o = p.Pointer.offset in
  let ok_perm =
    if write then Mir_view.writable p.Pointer.perm
    else Mir_view.readable p.Pointer.perm
  in
  let in_range =
    Int64.compare o p.Pointer.lo >= 0
    &&
    match Mir_layout.add o bytes with
    | Some e ->
        Int64.compare e p.Pointer.hi <= 0
        && Int64.compare e i.Instance.size <= 0
    | None -> false
  in
  let aligned = Mir_layout.aligned (Int64.add i.Instance.base o) ~align in
  if i.Instance.live && ok_perm && in_range && aligned then Ok i
  else Error Fault.Bad_access

(* Little-endian [bytes] at [p]: canonical zero-extended bits. *)
let load t p ~bytes ~align =
  match check t p ~bytes ~align ~write:false with
  | Error e -> Error e
  | Ok i ->
      let rec go k acc =
        if k < 0 then Ok acc
        else
          match get_byte i (Int64.add p.Pointer.offset (Int64.of_int k)) with
          | None -> Error Fault.Uninitialized
          | Some b ->
              go (k - 1) (Int64.logor (Int64.shift_left acc 8) (Int64.of_int b))
      in
      go (Int64.to_int bytes - 1) 0L

let store t p ~bytes ~align bits =
  match check t p ~bytes ~align ~write:true with
  | Error e -> Error e
  | Ok i ->
      for k = 0 to Int64.to_int bytes - 1 do
        set_byte i
          (Int64.add p.Pointer.offset (Int64.of_int k))
          (Int64.to_int
             (Int64.logand (Int64.shift_right_logical bits (8 * k)) 0xFFL))
      done;
      Ok ()

(* The defined bytes of [n] bytes from [offset], [None] for an undefined one. *)
let read_bytes t key ~offset ~n =
  let i = instance t key in
  Array.init n (fun k -> get_byte i (Int64.add offset (Int64.of_int k)))
