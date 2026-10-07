(* The id spaces of a Machine IR program. Each is a fresh type, so a value id
   cannot be passed where a block, object, helper or register unit is wanted. *)

module Block =
  Core.Tagged_int.Make
    (struct
      let prefix = "bb"
    end)
    ()

module Func =
  Core.Tagged_int.Make
    (struct
      let prefix = "fn"
    end)
    ()

module Helper =
  Core.Tagged_int.Make
    (struct
      let prefix = "helper"
    end)
    ()

module Instr =
  Core.Tagged_int.Make
    (struct
      let prefix = "i"
    end)
    ()

module Region =
  Core.Tagged_int.Make
    (struct
      let prefix = "region"
    end)
    ()

module Revision =
  Core.Tagged_int.Make
    (struct
      let prefix = "rev"
    end)
    ()

(* A runtime failure-site table entry: only the kinds whose record schema holds
   a site word index one. *)
module Site =
  Core.Tagged_int.Make
    (struct
      let prefix = "site"
    end)
    ()

(* A frame object before layout: a spill slot, local, save or scratch area. *)
module Slot =
  Core.Tagged_int.Make
    (struct
      let prefix = "slot"
    end)
    ()

(* One indivisible piece of register state; overlapping views share units. *)
module Unit =
  Core.Tagged_int.Make
    (struct
      let prefix = "u"
    end)
    ()

module Value =
  Core.Tagged_int.Make
    (struct
      let prefix = "%"
    end)
    ()

module View =
  Core.Tagged_int.Make
    (struct
      let prefix = "view"
    end)
    ()
