import {Board, Faction, PRE_OWNED} from "../Board.ts";
import {Consumable, KindDescriptions, Use} from "../components/Powerup.tsx";
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
    Donor,
    Influencer,
    Track,
    PoliceBrutality,
    Intimidate,
    VP,
    Rally,
    HQ,
    Donation,
    Reroll,
    Cheat,
    FreeSpeech,
    FiveYearPlan,
    //
    UnityLeft,
    UnityCenterLEFTRIGHT,
    UnityRight,
    UnityAuth,
    UnityCenterAUTHLIB,
    UnityLib,
    UnityAll
}

const consumableBgs = {
    interact: '#3f51b5',
    dyn: '#673ab7',
    base: '#444'
};

export const Consumables: Record<ConsumableTypes, Consumable> = {
    [ConsumableTypes.FreeSpeech]: new Consumable({
        name: "Free Speech",
        icon: "🔊",
        description: "Draw 3 cards (without changing your hand size)",
        onAction: (cell, _board) => {
        },
        use: Use.EXHAUST,
        cost: 1,
        factions: [Faction.CENTR, Faction.LIB, Faction.WILDCARD],
    }),
    [ConsumableTypes.FiveYearPlan]: new Consumable({
        name: "Five Year Plan",
        icon: "⭐",
        description: "Gain 6 Manuevers in next round",
        onAction: (cell, _board) => {
        },
        cost: 1,
        use: Use.SINGLE,
        factions: [Faction.COMM, Faction.SOC, Faction.NAT]
    }),
    [ConsumableTypes.Cheat]: new Consumable({
        name: "Cheat",
        icon: "😈",
        description: "Cheat in elections",
        onAction: (cell, _board) => {
        },
        cost: 4,
        use: Use.SINGLE,
        factions: [Faction.FASH, Faction.COMM, Faction.NAT]
    }),
    [ConsumableTypes.Reroll]: new Consumable({
        name: "Reshuffle",
        icon: "🔀",
        description: "Reshuffle Deck",
        onAction: (cell, board) => {
            // TODO reroll functionality
        },
        cost: 0,
        use: Use.FOREVER,
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
        cost: 3,
        use: Use.SINGLE,
        boardInteraction: true,
        bg: consumableBgs.dyn,
        factions: [Faction.CENTR, Faction.LIB, Faction.SOC, Faction.NAT]
    }),
    [ConsumableTypes.UnityLeft]: new Consumable({
        name: "Leftist Unity",
        icon: "❤️",
        description: "Expand {GREEN} {COMM} {SOC}",
        onAction: (cell, board, update, next) => {
            console.log('>>');
            board.doPopulism([Faction.GREEN, Faction.SOC, Faction.COMM], update, next).then();
        },
        cost: 5,
        use: Use.EXHAUST,
        bg: "linear-gradient(in srgb to right, var(--faction-COMM) 5%, var(--faction-SOC) 50%, var(--faction-GREEN) 95%)",
        factions: FactionLEFT
    }),
    [ConsumableTypes.UnityRight]: new Consumable({
        name: "Righwing Unity",
        icon: "💙",
        description: "Expand {FASH} {CON} {WILDCARD}",
        onAction: (cell, board, update, next) => {
            board.doPopulism([Faction.FASH, Faction.CON, Faction.WILDCARD], update, next).then();
        },
        cost: 5,
        use: Use.EXHAUST,
        bg: "linear-gradient(in srgb to right, var(--faction-FASH) 5%, var(--faction-CON) 50%, var(--faction-WILDCARD) 95%)",

        factions: FactionRIGHT
    }),
    [ConsumableTypes.UnityCenterLEFTRIGHT]: new Consumable({
        name: "Centrist Unity",
        icon: "💗",
        description: "Expand {NAT} {CENTR} {LIB}",
        onAction: (cell, board, update, next) => {
            board.doPopulism([Faction.NAT, Faction.CENTR, Faction.LIB], update, next).then();
        },
        bg: "linear-gradient(in srgb to right, var(--faction-NAT) 5%, var(--faction-CENTR) 50%, var(--faction-LIB) 95%)",
        cost: 5,
        use: Use.EXHAUST,
        factions: FactionC_RIGHT_LEFT
    }),
    [ConsumableTypes.UnityAuth]: new Consumable({
        name: "Auth. Unity",
        icon: "🖤",
        description: "Expand {COMM} {NAT} {FASH}",
        onAction: (cell, board, update, next) => {
            board.doPopulism([Faction.COMM, Faction.NAT, Faction.FASH], update, next).then();
        },
        cost: 5,
        use: Use.EXHAUST,
        bg: "linear-gradient(in srgb to right, var(--faction-COMM) 5%, var(--faction-NAT) 50%, var(--faction-FASH) 95%)",

        factions: FactionAUTH
    }),
    [ConsumableTypes.UnityCenterAUTHLIB]: new Consumable({
        name: "Centrist Unity",
        icon: "💓",
        description: "Expand {SOC} {CENTR} {CON}",
        onAction: (cell, board, update, next) => {
            board.doPopulism([Faction.SOC, Faction.CENTR, Faction.CON], update, next).then();
        },
        use: Use.EXHAUST,
        cost: 5,
        bg: "linear-gradient(in srgb to right, var(--faction-SOC) 5%, var(--faction-CENTR) 50%, var(--faction-CON) 95%)",
        factions: FactionC_AUTH_LIB
    }),
    [ConsumableTypes.UnityLib]: new Consumable({
        name: "Liberal Unity",
        icon: "💛",
        description: "Expand {GREEN} {LIB} {WILDCARD}",
        onAction: (cell, board, update, next) => {
            board.doPopulism([Faction.GREEN, Faction.LIB, Faction.WILDCARD], update, next).then();
        },
        cost: 5,
        use: Use.EXHAUST,
        bg: "linear-gradient(in srgb to right, var(--faction-GREEN) 5%, var(--faction-LIB) 50%, var(--faction-WILDCARD) 95%)",

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
        cost: 3,
        use: Use.EXHAUST,
        factions: FactionLEFT,
        bg: consumableBgs.interact
    }),
    [ConsumableTypes.PoliceBrutality]: new Consumable({
        name: "police brutality",
        icon: "👮",
        description: "Police Brutality :)",
        onAction: (cell, _board) => {
            //TODO
        },
        boardInteraction: true,
        cost: 3,
        use: Use.EXHAUST,
        factions: [Faction.NAT, Faction.FASH, Faction.CON],
        bg: consumableBgs.interact
    }),
    [ConsumableTypes.Follower]:
        new Consumable({
            name: "recruit ultra",
            icon: "🌈",
            description: KindDescriptions[Kind.RAINBOW],
            onAction:
                (cell, _board) => {
                    cell.kind = Kind.RAINBOW;
                },
            boardInteraction:
                true,
            cost: 2,
            factions: FactionALL,
            use: Use.EXHAUST,
            bg: consumableBgs.interact
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
            cost: 6,
            use: Use.SINGLE,
            factions: [Faction.SOC, Faction.CENTR, Faction.GREEN],
            bg: consumableBgs.dyn
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
                            if (!cell.owned && shouldJoin && cell.canCapture) {
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
            cost: 5,
            use: Use.SINGLE,
            factions: [Faction.SOC, Faction.FASH, Faction.CON, Faction.GREEN],
            bg: consumableBgs.dyn
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
            cost: 1,
            use: Use.EXHAUST,
            factions: [Faction.CON, Faction.FASH, Faction.FAITH],
            bg: consumableBgs.interact
        }),
    [ConsumableTypes.Influencer]:
        new Consumable({
            name: "recruit celebrity",
            icon: "🤩",
            description: KindDescriptions[Kind.INFLUENCER],
            onAction: (cell, _board) => {
                cell.kind = Kind.INFLUENCER;
            },
            boardInteraction: true,
            cost: 1,
            use: Use.EXHAUST,
            factions: [Faction.SOC, Faction.CENTR, Faction.CON],
            bg: consumableBgs.interact
        }),
    [ConsumableTypes.Donation]:
        new Consumable({
            name: "Donation",
            icon: "💷",
            description: "Arrange an anonymous donation. Earn 5 maneuvers.",
            onAction: (_cell, board) => {
                board.moves -= 5;
            },
            boardInteraction: false,
            cost: 0,
            use: Use.SINGLE,
            factions: [Faction.CENTR, Faction.LIB, Faction.WILDCARD],
        }),
    [ConsumableTypes.Track]:
        new Consumable({
            description: "Track Citizen, select a single tie to 'Keep' between rounds",
            icon: "🖲️",
            name: "Track Citizen",
            cost: 0,
            onAction(cell: Cell | undefined, board: Board, _update: React.Dispatch<React.SetStateAction<number>>, _nextStep: () => void): void {
                cell?.restore();
                cell.track = true;
                board.run.tracked.push(cell as Cell);
            },
            boardInteraction: true,
            bg: consumableBgs.interact,
            use: Use.EXHAUST,
            factions: [Faction.NAT, Faction.LIB, Faction.COMM, Faction.WILDCARD]
        }),
    [ConsumableTypes.Intimidate]:
        new Consumable({
            description: "Intimidate Citizen. They will vote for you out of fear for their life.",
            icon: "🔫",
            name: "intimidate",
            cost: 1,
            onAction(cell: Cell | undefined, board: Board, _update: React.Dispatch<React.SetStateAction<number>>, _nextStep: () => void): void {
                cell.owned = true;
            },
            boardInteraction: true,
            factions: FactionAUTH,
            use: Use.EXHAUST,
            bg: consumableBgs.interact
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