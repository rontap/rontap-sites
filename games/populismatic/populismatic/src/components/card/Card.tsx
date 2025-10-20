import './card.css';
import {Faction, RGBC} from "../../Factions.ts";
import {int} from "../../modifiers.ts";

type CardProps = {
    icon: string,
    children: JSX.Element,
    onClick: () => any,
    bg: RGBC,
    title: string,
    total: number,
    nth: number,
    toggle: () => any,
    selected: boolean,
    faction: Faction[]
    cost: int,
    kind: string,
    standablone: boolean
}
export default function Card({
                                 faction = [],
                                 selected,
                                 icon,
                                 toggle,
                                 children,
                                 onClick,
                                 bg,
                                 title,
                                 nth,
                                 total,
                                 cost,
                                 kind,
                                 standalone
                             }: CardProps) {

    const offset = (Math.abs(Math.floor(total / 2) - nth) * 13) + "px";
    const totalRounted = Math.ceil((total - 1) / 2);

    const radius = 300; // distance from circle center
    const angleStep = 8; // degrees between cards
    const middle = (total - 1) / 2;
    const centerYOffset = 0;


    const angle = (nth - middle) * angleStep;
    const x = radius * Math.sin(angle * Math.PI / 180);
    const y = radius * (1 - Math.cos(angle * Math.PI / 180));

    const transform = standalone ? "" : `
      translate(${2}px, ${y * 4}px) 
      rotate(${angle}deg)
    `;

    const alt = bg.startsWith("linear-gradient")
    return <div className={`cardOuter ${selected ? "cardToggled" : ""} ${alt ? "alt" : ""}`}
                style={{
                    "--card-bg-alt": bg,
                    "--card-bg": bg,
                    "--nth-card": nth,
                    "--total-card": totalRounted,
                    // "--offset": offset,
                    transform
                }}>

        <div className={`card card-kind-${kind}`} onClick={toggle ? (selected ? toggle : onClick) : onClick}>
            <div className={"cardCost"}>
                {cost}
            </div>
            <div className={"cardHeader"}>
                <div className={"cardTitle"}>{title}</div>
            </div>
            <div className={"cardContent"}>
                <div className={"cardImage"}>

                    <div className={"icon"}>
                        {icon}
                    </div>
                </div>
                {children}
            </div>
            <div className={"cardLower"}>
                {kind == "0" && "1️⃣"}
                {kind == "2" && "♾️"}
                {kind == "0" && "Single-Use"}
                {kind == "1" && "Use once per level"}
                {kind == "2" && "Eternal"}
            </div>

            <div className={"minipolcom"}>
                <div className={faction.filter(el => el == Faction.COMM).length ? "active" : "inactive"}></div>
                <div className={faction.filter(el => el == Faction.NAT).length ? "active" : "inactive"}></div>
                <div className={faction.filter(el => el == Faction.FASH).length ? "active" : "inactive"}></div>
                <div className={faction.filter(el => el == Faction.SOC).length ? "active" : "inactive"}></div>
                <div className={faction.filter(el => el == Faction.CENTR).length ? "active" : "inactive"}></div>
                <div className={faction.filter(el => el == Faction.CON).length ? "active" : "inactive"}></div>
                <div className={faction.filter(el => el == Faction.GREEN).length ? "active" : "inactive"}></div>
                <div className={faction.filter(el => el == Faction.LIB).length ? "active" : "inactive"}></div>
                <div
                    className={faction.find(el => el == Faction.WILDCARD || el == Faction.FAITH) ? "active" : "inactive"}></div>
            </div>
        </div>
    </div>
}