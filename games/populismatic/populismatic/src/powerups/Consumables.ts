import {Board, Faction, PRE_OWNED} from "../Board.ts";
import {Consumable, KindDescriptions} from "../components/Powerup.tsx";
import {Cell, Kind} from "../Cell.ts";
import React from "react";
import {
    FactionALL,
    FactionAUTH,
    FactionC_AUTH_LIB,
    FactionC_RIGHT_LEFT,
    FactionLEFT,
    FactionLIB,
    FactionRIGHT
} from "../Factions.ts";

export enum ConsumableTypes {
    Activist,
    Follower,
    VP,
    Rally,
    Donor,
    Influencer,
    Donation,
    Track,
    Intimidate,
    PoliceBrutality,
    HQ,
    Reroll,
    Cheat,
    //
    UnityLeft,
    UnityCenterLEFTRIGHT,
    UnityRight,
    UnityAuth,
    UnityCenterAUTHLIB,
    UnityLib,
    UnityAll
}

export const Consumables: Record<ConsumableTypes, Consumable> = {
    [ConsumableTypes.Cheat]: new Consumable({
        name: "Cheat",
        icon: "😈",
        description: "Cheat in elections",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: [Faction.FASH, Faction.COMM, Faction.NAT]
    }),
    [ConsumableTypes.Reroll]: new Consumable({
        name: "Reshuffle",
        icon: "🔀",
        description: "Reshuffle Deck",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: [Faction.CENTR, Faction.CON, Faction.WILDCARD, Faction.SOC, Faction.COMM]
    }),
    [ConsumableTypes.HQ]: new Consumable({
        name: "Move HQ",
        icon: "🛖",
        description: "Move Headquarters",
        onAction: (cell: Cell, board: Board) => {
            board.grid[board.origin[0]][board.origin[1]].isSource = false;
            board.origin = [cell.h, cell.w];
            board.grid[cell.h][cell.w].isSource = true;
            return;
        },
        cost: 60,
        boardInteraction: true,
        factions: [Faction.CENTR, Faction.LIB, Faction.SOC, Faction.NAT]
    }),
    [ConsumableTypes.UnityLeft]: new Consumable({
        name: "Leftist Unity",
        icon: "❤️",
        description: "Leftist Unity",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: FactionLEFT
    }),
    [ConsumableTypes.UnityRight]: new Consumable({
        name: "Righwing Unity",
        icon: "💙",
        description: "Righwing Unity",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: FactionRIGHT
    }),
    [ConsumableTypes.UnityCenterLEFTRIGHT]: new Consumable({
        name: "Centrist Unity",
        icon: "💗",
        description: "Centrist Unity",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: FactionC_RIGHT_LEFT
    }),
    [ConsumableTypes.UnityAuth]: new Consumable({
        name: "Authoritarian Unity",
        icon: "🖤",
        description: "Authoritarian Unity",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: FactionAUTH
    }),
    [ConsumableTypes.UnityCenterAUTHLIB]: new Consumable({
        name: "Centrist Unity",
        icon: "💓",
        description: "Centrist Unity",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: FactionC_AUTH_LIB
    }),
    [ConsumableTypes.UnityLib]: new Consumable({
        name: "Liberal Unity",
        icon: "💛",
        description: "Liberal Unity",
        onAction: (cell, _board) => {
        },
        cost: 60,
        factions: FactionLIB
    }),

    [ConsumableTypes.Activist]: new Consumable({
        name: "Recruit Activist",
        icon: "💣",
        description: KindDescriptions[Kind.ACTIVIST],
        onAction: (cell, _board) => {
            cell.kind = Kind.ACTIVIST;
        },
        boardInteraction: true,
        cost: 60,
        factions: FactionLEFT
    }),
    [ConsumableTypes.PoliceBrutality]: new Consumable({
        name: "police brutality",
        icon: "👮",
        description: "Police Brutality :)",
        onAction: (cell, _board) => {

        },
        boardInteraction: true,
        cost: 60,
        factions: [Faction.NAT, Faction.FASH, Faction.CON]
    }),
    [ConsumableTypes.Follower]:
        new Consumable({
            name: "Dedicated Follower",
            icon: "🌈",
            description: KindDescriptions[Kind.RAINBOW],
            onAction:
                (cell, _board) => {
                    cell.kind = Kind.RAINBOW;
                },
            boardInteraction:
                true,
            cost: 60,
            factions: FactionALL
        }),

    [ConsumableTypes.VP]:
        new Consumable({
            name: "VP",
            icon: "🧑‍🤝‍🧑",
            description: "Pick a VP. They will be able to campaign for you at a different part of the country, changing ideologies as you do.",
            onAction: (cell, _board) => {
                cell.source = true;
            },
            boardInteraction: true,
            cost: 180,

            factions: [Faction.SOC, Faction.CENTR, Faction.GREEN]
        }),
    [ConsumableTypes.Rally]:
        new Consumable({
            name: "Hold Rally",
            icon: "🏟️",
            description: "Hold a rally. People at the periphery of your control may join your ideology.",
            onAction: (_cell, board, update, nextStage) => {
                board.forEach(baseCell => {
                    if (baseCell.owned) {
                        baseCell.eachNeighbour().forEach(cell => {
                            const shouldJoin = Math.random() > 0.33;
                            if (!cell.owned && shouldJoin) {
                                cell.owned = PRE_OWNED;
                                cell.faction = board.getOrigin.faction;
                            }
                        })
                    }
                })
                setTimeout(() => {
                    board.doPopulism(board.getOrigin.faction, update, nextStage).then(() => false);
                }, 400)
            },
            boardInteraction: false,
            cost: 150,
            factions: [Faction.SOC, Faction.FASH, Faction.CON, Faction.GREEN]
        }),
    [ConsumableTypes.Donor]:
        new Consumable({
            name: "Recruit Donor",
            icon: "📈",
            description: "Recruit a voter as a donor. When capturing it, it will give you extra influence.",
            onAction: (_cell, board) => {
                const addMoves = 10;
                board.run.influence += addMoves;
            },
            boardInteraction: true,
            cost: 80,
            factions: [Faction.CON, Faction.FASH, Faction.FAITH]
        }),
    [ConsumableTypes.Influencer]:
        new Consumable({
            name: "Recruit Influencer",
            icon: "🤩",
            description: KindDescriptions[Kind.INFLUENCER],
            onAction: (cell, _board) => {
                cell.kind = Kind.INFLUENCER;
            },
            boardInteraction: true,
            cost: 60,
            factions: [Faction.SOC, Faction.CENTR, Faction.CON]
        }),
    [ConsumableTypes.Donation]:
        new Consumable({
            name: "Anonymus Donation",
            icon: "💷",
            description: "Arrange an anonymous donation. Earn 5 maneuvers.",
            onAction: (_cell, board) => {
                board.moves -= 5;
            },
            boardInteraction: false,
            cost: 60,
            factions: [Faction.CENTR, Faction.LIB, Faction.WILDCARD]
        }),
    [ConsumableTypes.Track]:
        new Consumable({
            description: "Track Citzen, select a single tie to 'Keep' between rounds",
            icon: "🖲️",
            name: "Track Citizen",
            cost: 60,
            onAction(cell: Cell | undefined, board: Board, _update: React.Dispatch<React.SetStateAction<number>>, _nextStep: () => void): void {
                cell?.restore();
                cell.track = true;
                board.run.tracked.push(cell as Cell);
            },
            boardInteraction: true,
            factions: [Faction.NAT, Faction.LIB, Faction.COMM, Faction.WILDCARD]
        }),
    [ConsumableTypes.Intimidate]:
        new Consumable({
            description: "Intimidate Citizen. They will vote for you out of fear for their life.",
            icon: "🔫",
            name: "intimidate",
            cost: 60,
            onAction(cell: Cell | undefined, board: Board, _update: React.Dispatch<React.SetStateAction<number>>, _nextStep: () => void): void {
                cell.owned = true;
            },
            boardInteraction: true,
            factions: FactionAUTH
        }),
// new Consumable({
//     cost: 40,
//     description: "Use gerrymandering techniques to lower your needed control by -1%. Increases disenfrenchised voter weight.",
//     icon: "",
//     name: "Gerrymander",
//     onAction(_cell: Cell | undefined, board: Board, _update: React.Dispatch<React.SetStateAction<number>>, _nextStep: () => void): void {
//         board.run.modifiers.winConditions.required -= 1;
//     }
// })
// new Consumable({
//     name: "Disenfranchised Voter",
//     icon: "🤷",
//     description: KindDescriptions[Kind.DISENFRANCHISED],
//     onAction: (cell, _board) => {
//         cell.kind = Kind.DISENFRANCHISED;
//     },
//     boardInteraction: true,
//     cost: 20
// }),
// new Consumable({
//     name: "Tactical Voter",
//     icon: "🤡",
//     description: KindDescriptions[Kind.TACTICAL],
//     onAction: (cell, _board) => {
//         cell.kind = Kind.TACTICAL;
//     },
//     boardInteraction: true,
//     cost: 20
// }),

} as const;